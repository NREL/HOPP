import datetime as dt
import matplotlib.pyplot as plt
import math as m
import numpy as np
import matplotlib.dates as dates

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return m.ceil(n * multiplier) / multiplier

def roundup_to_base(x, base=5):
    return base * round_up(float(x)/base)

def gen_plot(dismod, is_battery = True,title = None, grid_check = []):
    '''
    This function creates stacked barplot of generation over a time horizon
        by source for a PV-CSP-battery system.
    
    Plotting with battery charging and discharging could use some work... 
    '''
    st_day = 0
    N_days = int(len(dismod.wdotS[:]())*dismod.Delta()/24)
    min_step = int(dismod.Delta()*60)
    pow_scale = 1/1000. # convert from kW to MW
    #start time
    dfst = dt.datetime(2019, 1, 1, 0, int(min_step/2))
    index = [dfst + dt.timedelta(minutes=i*min_step) for i in range(len(dismod.wdotS[:]()))]
    
    ##---------- Cleaning up data ------------------
    wbc = [-x*pow_scale for x in dismod.wdotBC[:]()]    
    wbd = [x*pow_scale for x in dismod.wdotBD[:]()]

    wpv = [x*pow_scale for x in dismod.wdotPV[:]()]
    wpv_curtail = [(pvR - pvA)*pow_scale for pvR,pvA in zip(dismod.Wpv[:](),dismod.wdotPV[:]())]
    # TODO:could add clipping if we start modeling DC side

    wwf = [x*pow_scale for x in dismod.wdotWF[:]()]
    wwf_curtail = [(wfR - wfA)*pow_scale for wfR,wfA in zip(dismod.Wwf[:](),dismod.wdotWF[:]())]
    
    wpur = [x*pow_scale for x in dismod.wdotP[:]()]
    wgrid = [x*pow_scale for x in dismod.wdotS[:]()]

    soc = dismod.bsoc[:]()    
    # Price = dismod.P[:]()
    Price = [x*10. for x in dismod.P[:]()]

    
    ##--------- Time range -----------------------
    dfst = dt.datetime(2019, 1, 1, 0, 0)
    st = (60/min_step)*24*st_day
    en = st+len(wgrid)
    dst = dfst + dt.timedelta(minutes=st*min_step)
    den = dfst + dt.timedelta(minutes=en*min_step)
    #mask = (index >= dst) & (index < den)
    
    ##------- Plotting data ------------------------
    axis_fsize = 28
    axis_tick_fsize = 20
    
    plt.figure(figsize=(12,5.5))
    width = dt.timedelta(minutes=min_step).total_seconds()/dt.timedelta(days=1).total_seconds()
    #plt.bar(index[mask], wwf[mask], width, color = 'gray', edgecolor = 'gray', label=r"CSP Power Cycle")
    plt.bar(index, wwf, width, color = 'blue', edgecolor = 'white', label=r"Wind Farm", alpha = 0.7)
    
    bot = wwf
    plt.bar(index, wpv, width, bottom = bot, color = 'orange', edgecolor = 'white', label=r"PV Field", alpha = 0.7)
    bot = [b+pv for b,pv in zip(bot, wpv)]

    if is_battery:
        plt.bar(index, wbd, width, bottom = bot, color = 'cyan', edgecolor = 'white', label=r"Battery Discharge", alpha = 0.7)
        plt.bar(index, wbc,width, color='green', edgecolor='white', label=r"Battery Charge", alpha = 0.7)
        bot = [b+bat for b,bat in zip(bot, wbd)]
    
    plt.bar(index, [wfc + pvc for wfc, pvc in zip(wwf_curtail, wpv_curtail)], width, bottom = bot, color = 'red', edgecolor = 'white', label=r"Energy Curtailment", alpha = 0.7)    
    
    if is_battery:
        plt.legend(loc='upper left',ncol =2, prop = {'size': axis_tick_fsize, 'weight': 'bold'})
        
        ymin = min(wbc)  # these are both negative values
        ymin = -int(roundup_to_base(-ymin)) - 5
        
        ymax = max(wwf + wpv + wbd)
        ymax = int(roundup_to_base(ymax)) + 2*5
        plt.ylim(ymin, ymax) #650
        #plt.ylim(-15, 15) #650
    else:
        plt.legend(loc='upper left', prop = {'size': axis_tick_fsize, 'weight': 'bold'})
        ymin = 0
        
        ymax = max(wwf + wpv)
        ymax = int(roundup_to_base(ymax)) + 5 
        plt.ylim(ymin,ymax) #650
        
        #plt.ylim(0,200) #650
    
    
    plt.yticks(fontsize = axis_tick_fsize, fontweight='bold')
    plt.xticks(fontsize = axis_tick_fsize, fontweight='bold')
    #plt.xlabel("Time", fontsize = axis_fsize, fontweight = 'bold')

    ax1 = plt.gca()
    if len(grid_check) > 1:
        ax1.plot(index, grid_check, color = 'pink', linewidth = 2.0, label ="Grid")
        ax1.plot(index, wgrid, color = 'k', linewidth = 2.0, label ="Grid")
    
    ax2 = ax1.twinx()
    ax2.plot(index, Price, color = 'k', linewidth = 3.0, label = r"PPA Mult.")
    if is_battery:
        ax2.plot(index, soc, color = 'purple', linewidth = 3.0, label = "SOC")
        
        # ymin = -0.5
        # ymax = max(max(Price),max(soc))
        # ymax = roundup_to_base(ymax,0.5) + 0.5
        # plt.ylim(ymin,ymax)      #Summer
        # yticks = ax2.yaxis.get_major_ticks()
        # yticks[0].set_visible(False)
        plt.ylim(-1.0,2.0)
    else:
        ymin = 0.0
        ymax = max(Price)#,max(df.soc_s[mask]))
        ymax = roundup_to_base(ymax,0.5) + 0.5
        #plt.ylim(ymin,ymax)      #SummerS
        plt.ylim(0.0,2.5)      #Summer     

    wSold = [x/max(dismod.wdotS[:]()) for x in dismod.wdotS[:]()]
    ax2.plot(index, wSold, color = 'g', linewidth = 3.0, label = r"Net Prod.")   
    plt.legend(loc='upper right', prop = {'size': axis_tick_fsize, 'weight': 'bold'})
    
    ## sets the step size of y1 axis to 50 MW
    #start, end = ax2.get_ylim()
    #ax2.yaxis.set_ticks(np.arange(start,end, 0.5))
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start,end + 5, 5))
    
    #align_yaxis(ax1, 0, ax2, 0)     # aligns zeros of the two y-axis
    
    # aligns gridlines for the two y-axis
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    
    ax1.grid(ls = ':')
    ax1.set_ylabel("Gross Power [MWe]", fontsize=axis_fsize, fontweight='bold')

    if is_battery:
        ax2.set_ylabel("SOC [-], PPA Mult. [-]", fontsize = axis_fsize, fontweight ='bold')
    else:
        ax2.set_ylabel("PPA Mult. [-]", fontsize = axis_fsize, fontweight ='bold')
    plt.yticks(fontsize = axis_tick_fsize, fontweight='bold')
    
    #plt.gca().xaxis.set_ticks(np.arange(st,en, 1))
    
    ## Formatting axis labels
    arr = np.array([dst + dt.timedelta(days=i) for i in range(N_days+1)])
    ax1.set_xticks(arr)
    plt.xlim(dst,den)
    ax1.tick_params(axis='x', which = 'major', pad = 10)
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    
    if not title == None:
        plt.title(title, fontsize = 24, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    return

def gen_subplot(dismod, is_battery = True,title = None, grid_check = []):
    #TODO: subplots
    '''
    This function creates stacked barplot of generation over a time horizon
        by source for a PV-CSP-battery system.
    
    Plotting with battery charging and discharging could use some work... 
    '''
    st_day = 0
    N_days = int(len(dismod.wdotS[:]())*dismod.Delta()/24)
    min_step = int(dismod.Delta()*60)
    pow_scale = 1/1000. # convert from kW to MW
    #start time
    dfst = dt.datetime(2019, 1, 1, 0, int(min_step/2))
    index = [dfst + dt.timedelta(minutes=i*min_step) for i in range(len(dismod.wdotS[:]()))]
    
    ##---------- Cleaning up data ------------------
    wbc = [-x*pow_scale for x in dismod.wdotBC[:]()]    
    wbd = [x*pow_scale for x in dismod.wdotBD[:]()]

    wpv = [x*pow_scale for x in dismod.wdotPV[:]()]
    wpv_curtail = [(pvR - pvA)*pow_scale for pvR,pvA in zip(dismod.Wpv[:](),dismod.wdotPV[:]())]
    # TODO:could add clipping if we start modeling DC side

    wwf = [x*pow_scale for x in dismod.wdotWF[:]()]
    wwf_curtail = [(wfR - wfA)*pow_scale for wfR,wfA in zip(dismod.Wwf[:](),dismod.wdotWF[:]())]
    
    wpur = [x*pow_scale for x in dismod.wdotP[:]()]
    wgrid = [x*pow_scale for x in dismod.wdotS[:]()]

    soc = dismod.bsoc[:]()    
    # Price = dismod.P[:]()
    Price = [x*10. for x in dismod.P[:]()]

    
    ##--------- Time range -----------------------
    dfst = dt.datetime(2019, 1, 1, 0, 0)
    st = (60/min_step)*24*st_day
    en = st+len(wgrid)
    dst = dfst + dt.timedelta(minutes=st*min_step)
    den = dfst + dt.timedelta(minutes=en*min_step)
    #mask = (index >= dst) & (index < den)
    
    ##------- Plotting data ------------------------
    axis_fsize = 28
    axis_tick_fsize = 20
    
    plt.figure(figsize=(12,5.5))
    width = dt.timedelta(minutes=min_step).total_seconds()/dt.timedelta(days=1).total_seconds()
    #plt.bar(index[mask], wwf[mask], width, color = 'gray', edgecolor = 'gray', label=r"CSP Power Cycle")
    plt.bar(index, wwf, width, color = 'blue', edgecolor = 'white', label=r"Wind Farm", alpha = 0.7)
    
    bot = wwf
    plt.bar(index, wpv, width, bottom = bot, color = 'orange', edgecolor = 'white', label=r"PV Field", alpha = 0.7)
    bot = [b+pv for b,pv in zip(bot, wpv)]

    if is_battery:
        plt.bar(index, wbd, width, bottom = bot, color = 'cyan', edgecolor = 'white', label=r"Battery Discharge", alpha = 0.7)
        plt.bar(index, wbc,width, color='green', edgecolor='white', label=r"Battery Charge", alpha = 0.7)
        bot = [b+bat for b,bat in zip(bot, wbd)]
    
    plt.bar(index, [wfc + pvc for wfc, pvc in zip(wwf_curtail, wpv_curtail)], width, bottom = bot, color = 'red', edgecolor = 'white', label=r"Energy Curtailment", alpha = 0.7)    
    
    if is_battery:
        plt.legend(loc='upper left',ncol =2, prop = {'size': axis_tick_fsize, 'weight': 'bold'})
        
        ymin = min(wbc)  # these are both negative values
        ymin = -int(roundup_to_base(-ymin)) - 5
        
        ymax = max(wwf + wpv + wbd)
        ymax = int(roundup_to_base(ymax)) + 2*5
        plt.ylim(ymin, ymax) #650
        #plt.ylim(-15, 15) #650
    else:
        plt.legend(loc='upper left', prop = {'size': axis_tick_fsize, 'weight': 'bold'})
        ymin = 0
        
        ymax = max(wwf + wpv)
        ymax = int(roundup_to_base(ymax)) + 5 
        plt.ylim(ymin,ymax) #650
        
        #plt.ylim(0,200) #650
    
    
    plt.yticks(fontsize = axis_tick_fsize, fontweight='bold')
    plt.xticks(fontsize = axis_tick_fsize, fontweight='bold')
    #plt.xlabel("Time", fontsize = axis_fsize, fontweight = 'bold')

    ax1 = plt.gca()
    if len(grid_check) > 1:
        ax1.plot(index, grid_check, color = 'pink', linewidth = 2.0, label ="Grid")
        ax1.plot(index, wgrid, color = 'k', linewidth = 2.0, label ="Grid")
    
    ax2 = ax1.twinx()
    ax2.plot(index, Price, color = 'k', linewidth = 3.0, label = r"PPA Mult.")
    if is_battery:
        ax2.plot(index, soc, color = 'purple', linewidth = 3.0, label = "SOC")
        
        # ymin = -0.5
        # ymax = max(max(Price),max(soc))
        # ymax = roundup_to_base(ymax,0.5) + 0.5
        # plt.ylim(ymin,ymax)      #Summer
        # yticks = ax2.yaxis.get_major_ticks()
        # yticks[0].set_visible(False)
        plt.ylim(-1.0,2.0)
    else:
        ymin = 0.0
        ymax = max(Price)#,max(df.soc_s[mask]))
        ymax = roundup_to_base(ymax,0.5) + 0.5
        #plt.ylim(ymin,ymax)      #SummerS
        plt.ylim(0.0,2.5)      #Summer     

    wSold = [x/max(dismod.wdotS[:]()) for x in dismod.wdotS[:]()]
    ax2.plot(index, wSold, color = 'g', linewidth = 3.0, label = r"Net Prod.")   
    plt.legend(loc='upper right', prop = {'size': axis_tick_fsize, 'weight': 'bold'})
    
    ## sets the step size of y1 axis to 50 MW
    #start, end = ax2.get_ylim()
    #ax2.yaxis.set_ticks(np.arange(start,end, 0.5))
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start,end + 5, 5))
    
    #align_yaxis(ax1, 0, ax2, 0)     # aligns zeros of the two y-axis
    
    # aligns gridlines for the two y-axis
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    
    ax1.grid(ls = ':')
    ax1.set_ylabel("Gross Power [MWe]", fontsize=axis_fsize, fontweight='bold')

    if is_battery:
        ax2.set_ylabel("SOC [-], PPA Mult. [-]", fontsize = axis_fsize, fontweight ='bold')
    else:
        ax2.set_ylabel("PPA Mult. [-]", fontsize = axis_fsize, fontweight ='bold')
    plt.yticks(fontsize = axis_tick_fsize, fontweight='bold')
    
    #plt.gca().xaxis.set_ticks(np.arange(st,en, 1))
    
    ## Formatting axis labels
    arr = np.array([dst + dt.timedelta(days=i) for i in range(N_days+1)])
    ax1.set_xticks(arr)
    plt.xlim(dst,den)
    ax1.tick_params(axis='x', which = 'major', pad = 10)
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    
    if not title == None:
        plt.title(title, fontsize = 24, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    return