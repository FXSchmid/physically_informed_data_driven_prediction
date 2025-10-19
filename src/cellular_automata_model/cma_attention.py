
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

class CellularAutomataModel:
    def __init__(self, wd_current, dem, roughness, rainfall, cellsize=5, nodata=0):
        self.depths = wd_current
        self.dem = dem
        self.roughness = roughness
        self.rainfall = rainfall
        self.cellsize = cellsize
        
        self.mask = (self.dem == nodata)
        self.maskn = self.shift_raster(self.mask, -1, 0, True) | self.mask
        self.maske = self.shift_raster(self.mask, 0, -1, True) | self.mask
        self.masks = self.shift_raster(self.mask, 1, 0, True) | self.mask
        self.maskw = self.shift_raster(self.mask, 0, 1, True) | self.mask
        
        self.wse = np.where(self.mask, 0.0, self.dem + self.depths)
        
        self.intercellular_volume_current = np.where(self.mask, 0.0, self.depths*(self.cellsize**2))
        self.intercellular_volume_next = np.copy(self.intercellular_volume_current)
        
        self.g = 9.80665
    
    def run(self, start=0, end=5*60, dtfix=None, dtmin=None, dtmax=None, tau=0.001, sigma=0.0001):
        
        print("Starting Simulation")
        ts = datetime.now()
        time = start
        
        while time < end:
            
            ti = datetime.now()
            print("--------")
            
            # hydraulic gradients
            hgradn = self.calculate_hydraulic_gradients(self.wse, north=True)
            hgrade = self.calculate_hydraulic_gradients(self.wse, east=True)
            hgrads = self.calculate_hydraulic_gradients(self.wse, south=True)
            hgradw = self.calculate_hydraulic_gradients(self.wse, west=True)
            hgradmax = np.maximum.reduce([hgradn, hgrade, hgrads, hgradw])
            
            # velocities
            depths_masked = np.where(self.mask, 0.0, self.depths)
            roughness_masked = np.where(self.mask, 0.0, self.roughness)
            ucrt = np.sqrt(self.g*depths_masked)
            uman = (1/(roughness_masked + 1e-300))*(depths_masked**(2/3))*np.sqrt(abs(hgradmax/self.cellsize))
            umax = np.minimum.reduce([ucrt, uman])
            umaxmax = np.max(umax)
            print("\tMax Velocity: {:.3f} mps".format(umaxmax))
            
            # time step calculations
            if dtfix != None:
                dt = dtfix
                print("\t--Using Fixed dt--")
            else:
                dtc = self.calculate_time_step(hgradmax, sigma)
                dtcmax = np.min(dtc)
                print("\tCalculated dt: {:.3f} s".format(dtcmax))
                dt = dtcmax
                if (dtmin != None) and (dt < dtmin):
                    dt = dtmin
                elif (dtmax != None) and (dt > dtmax):
                    dt = dtmax
                if (time + dt) > end:
                    dt = end - time
            time += dt
            print("\tApplied dt: {:.3f} s".format(dt))
            print("\tSimulation Time: {:.3f} s".format(time))
            
            # calculate rainfall
            r = (self.rainfall/(1000*3600))*dt
            
            # volumetric gradients
            vgradn = hgradn*(self.cellsize**2)
            vgrade = hgrade*(self.cellsize**2)
            vgrads = hgrads*(self.cellsize**2)
            vgradw = hgradw*(self.cellsize**2)
            vgradntau = np.where(hgradn < tau, 0.0, hgradn)
            vgradetau = np.where(hgrade < tau, 0.0, hgrade)
            vgradstau = np.where(hgrads < tau, 0.0, hgrads)
            vgradwtau = np.where(hgradw < tau, 0.0, hgradw)
            vgradmin = np.minimum.reduce([vgradntau, vgradetau, vgradstau, vgradwtau])
            vgradtot = np.add.reduce([vgradn, vgrade, vgrads, vgradw])
            
            # weights
            wn = vgradn/(vgradtot + vgradmin + 1e-300)
            we = vgrade/(vgradtot + vgradmin + 1e-300)
            ws = vgrads/(vgradtot + vgradmin + 1e-300)
            ww = vgradw/(vgradtot + vgradmin + 1e-300)
            wmax = np.maximum.reduce([wn, we, ws, ww])
            
            # intercellular volumes
            ia = np.where(self.mask, 0.0, self.depths*(self.cellsize**2))
            ib = np.where(self.mask, 0.0, (umax*self.depths*self.cellsize*dt)/(wmax + 1e-300))
            ic = np.where(self.mask, 0.0, vgradmin + self.intercellular_volume_current)
            itot = np.minimum.reduce([ia, ib, ic])
            ivn = itot*wn
            ive = itot*we
            ivs = itot*ws
            ivw = itot*ww
            ivsum = np.add.reduce([ivn, ive, ivs, ivw])
            self.intercellular_volume_next = self.apply_intercellular_volumes(ivsum, rain=r)
            self.intercellular_volume_next = self.apply_intercellular_volumes(ivn, north=True)
            self.intercellular_volume_next = self.apply_intercellular_volumes(ive, east=True)
            self.intercellular_volume_next = self.apply_intercellular_volumes(ivs, south=True)
            self.intercellular_volume_next = self.apply_intercellular_volumes(ivw, west=True)
            
            # need these two lines to prevent depths going negative in the simulation
            self.intercellular_volume_next = np.nan_to_num(self.intercellular_volume_next)
            self.intercellular_volume_next = np.where(self.intercellular_volume_next < 0, 0.0, self.intercellular_volume_next)
            
            # apply new depths
            self.depths = np.where(self.mask, 0.0, self.intercellular_volume_next/(self.cellsize**2))
            self.wse = np.where(self.mask, 0.0, self.dem + self.depths)
            self.intercellular_volume_current = np.copy(self.intercellular_volume_next)
            
            tf = datetime.now()
            print("\tTime Step Solve Time:", tf - ti)
        
        te = datetime.now()
        print("--------")
        print("Total Solve Time:", te - ts)
        
        return self.depths
    
    def calculate_hydraulic_gradients(self, wse, north=False, east=False, south=False, west=False):
        dx, dy = 0, 0
        if north:
            dx, dy = -1, 0
            mask = self.maskn
        elif east:
            dx, dy = 0, -1
            mask = self.maske
        elif south:
            dx, dy = 1, 0
            mask = self.masks
        elif west:
            dx, dy = 0, 1
            mask = self.maskw
        wse_shifted = self.shift_raster(wse, dx, dy, 0.0)
        hgrad = np.where(mask, 0.0, wse - wse_shifted)
        hgrad = np.where(hgrad < 0, 0.0, hgrad)
        return hgrad
    
    def apply_intercellular_volumes(self, itot, north=False, east=False, south=False, west=False, rain=0):
        dx, dy = 0, 0
        if north:
            dx, dy = 1, 0
        elif east:
            dx, dy = 0, 1
        elif south:
            dx, dy = -1, 0
        elif west:
            dx, dy = 0, -1
        if dx == 0 and dy == 0:
            vol = np.where(self.mask, 0.0, self.intercellular_volume_next - itot + rain*(self.cellsize**2))
        else:
            itot_shifted = self.shift_raster(itot, dx, dy, 0.0)
            vol = np.where(self.mask, 0.0, self.intercellular_volume_next + itot_shifted)
        return vol
    
    def shift_raster(self, arr, shift_x, shift_y, fill=0):
        rows, cols = arr.shape
        shifted_arr = np.full_like(arr, fill)
        if shift_x > 0:
            start_x_src = 0
            end_x_src = rows - shift_x
            start_x_dst = shift_x
            end_x_dst = rows
        else:
            start_x_src = -shift_x
            end_x_src = rows
            start_x_dst = 0
            end_x_dst = rows + shift_x
    
        if shift_y > 0:
            start_y_src = 0
            end_y_src = cols - shift_y
            start_y_dst = shift_y
            end_y_dst = cols
        else:
            start_y_src = -shift_y
            end_y_src = cols
            start_y_dst = 0
            end_y_dst = cols + shift_y
        shifted_arr[start_x_dst:end_x_dst, start_y_dst:end_y_dst] = arr[start_x_src:end_x_src, start_y_src:end_y_src]
        return shifted_arr
    
    def calculate_time_step(self, hgradmax, sigma):
        depths_masked = np.where(self.mask, 0.0, self.depths)
        roughness_masked = np.where(self.mask, 0.0, self.roughness)
        s = abs(hgradmax/self.cellsize)
        mann = np.where(s < sigma, 1e30, (2*roughness_masked/(depths_masked**(5/3) + 1e-300))*np.sqrt(s))
        mannmin = np.min(mann)
        dt = (1/4)*(self.cellsize**2)*mannmin
        return dt

if __name__ == "__main__":
    
    path_0 = r"...." # here should be the current/initial water depth 
    path_1 = r"...." # here should be a potential true water depth for quality check of the CAM model 
    path_2 = r"..."  # here should be the DEM 
    path_3 = r" .."  # roughness values (same extent and resolution as DEM)

    wd_true = np.genfromtxt(path_1, skip_header=6)
    wd_current = np.genfromtxt(path_0, skip_header=6)

    rain = 99 # mm/hr
    dem = np.genfromtxt(path_2, skip_header=6)
    zero_mask = dem == 0
    roughness = np.genfromtxt(path_3, skip_header=6)

    
    model = CellularAutomataModel(wd_current=wd_current, dem=dem, roughness=roughness, rainfall=rain)
    CMA_attention_5min_water_depth = model.run(dtfix=30, dtmin=2, dtmax=10)

    CMA_attention_5min_water_depth = np.where(zero_mask, 0, CMA_attention_5min_water_depth)


    










