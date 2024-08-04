# TODO-1 Importing Tabula Parameter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


sfh = "C:/Users/elvis/Desktop/Pyclass/Thesis/Thesis_data_simulation/parameter_SFH.csv"
th = "C:/Users/elvis/Desktop/Pyclass/Thesis/Thesis_data_simulation/parameter_TH.csv"
mfh = "C:/Users/elvis/Desktop/Pyclass/Thesis/Thesis_data_simulation/parameter_MFH.csv"
ab = "C:/Users/elvis/Desktop/Pyclass/Thesis/Thesis_data_simulation/parameter_AB.csv"
sfh1 = pd.read_csv(sfh,skiprows=36,nrows=10, index_col=0)
th1 = pd.read_csv(th,skiprows=36,nrows=10, index_col=0)
mfh1 = pd.read_csv(mfh,skiprows=36,nrows=10, index_col=0)
ab1 = pd.read_csv(ab,skiprows=36,nrows=10, index_col=0)

# TODO - Session_2 : Weather Data
simulation_data = "C:/Users/elvis/Desktop/Heating Demand/working data.csv"
weather = pd.read_csv(simulation_data, skiprows=8, nrows=8760, index_col=0)
weather.index = pd.to_datetime(weather.index, format="%Y%m%d:%H%M")
weather.index = pd.date_range(start="2009-01-01 00:00", end="2009-12-31 23:00", freq="h")

# TODO - 2.1 Hourly Calaculation

# Tabula Data
teta = 20.0                              # indoor temp heating season
A_ref = [121.2, 117.4, 3129.1, 3887.4]   # reference area [m2]
btr_ex = 1.0                             # adjustment soil factor for construction bordering on external air
btr_exw = 0.0
btr_floor = 0.5                          # adjustment soil factor of unheated cellar/construction bordering on soil
Fnu = [0.80, 0.89, 0.93, 0.92]           # temp reduction (high thermal conductivity
base_temp = 12.0                         # Base temp [°C]
DU_tbr = 0.10

cp_air = 0.34              # Volume specific heat capacity of air            [Wh/(m^3.K)]
n_air_use = 0.40           # Air change rate by use                          [1/h]
n_air_inf = 0.20           # Air change rate by infiltration                 [1/h]
h_room = 2.5               # standard value of room height                   [m]

A_roof = [sfh1.iloc[0][0], th1.iloc[0][0], mfh1.iloc[0][0], ab1.iloc[0][0]]                       # roof envelope area [m2]
A_roof1 = [sfh1.iloc[1][0], th1.iloc[1][0], mfh1.iloc[1][0], ab1.iloc[1][0]]
U_roof = [sfh1.iloc[0][1], th1.iloc[0][15], mfh1.iloc[0][15], ab1.iloc[0][15]]     # actual u value [W/(m2K)]
U_roof1 = [sfh1.iloc[1][1], th1.iloc[1][15], mfh1.iloc[1][15], ab1.iloc[1][15]]    # actual u value [W/(m2K)]

A_wall1 = [sfh1.iloc[2][0], th1.iloc[2][0], mfh1.iloc[2][0], ab1.iloc[2][0]]     # Wall envelope area [m2]
A_wall2 = [sfh1.iloc[3][0], th1.iloc[3][0], mfh1.iloc[3][0], ab1.iloc[3][0]]     # Wall envelope area [m2]
U_wall1 = [sfh1.iloc[2][1], th1.iloc[2][15],mfh1.iloc[2][15], ab1.iloc[2][15]]     # Heat transfer coeff - actual u value [W/(m2K)]
U_wall2 = [sfh1.iloc[3][1], th1.iloc[3][15],mfh1.iloc[3][15], ab1.iloc[3][15]]     # Heat transfer coeff - actual u value [W/(m2K)]

A_floor = [sfh1.iloc[5][0], th1.iloc[5][0], mfh1.iloc[5][0], ab1.iloc[5][0]]      # floor envelope area [m2]
A_floor1 = [sfh1.iloc[-4][0], th1.iloc[-4][0], mfh1.iloc[-4][0], ab1.iloc[-4][0]] # floor envelope area [m2]
U_floor = [sfh1.iloc[5][1], th1.iloc[5][15],mfh1.iloc[5][15], ab1.iloc[5][15]]      # actual u value [W/(m2K)]
U_floor1 = [sfh1.iloc[6][1], th1.iloc[6][15], mfh1.iloc[6][15], ab1.iloc[6][15]]    # actual u value [W/(m2K)]

A_win = [sfh1.iloc[-3][0], th1.iloc[-3][0], mfh1.iloc[-3][0], ab1.iloc[-3][0]]    # window envelope area [m2]
A_win1 = [sfh1.iloc[-2][0], th1.iloc[-2][0], mfh1.iloc[-2][0], ab1.iloc[-2][0]]   # window envelope area [m2
U_win = [sfh1.iloc[-3][1], th1.iloc[-3][15],mfh1.iloc[-3][15], ab1.iloc[-3][15]]    # actual u value [W/(m2K)]
U_win1 = [sfh1.iloc[-2][1], th1.iloc[-2][15],mfh1.iloc[-2][15], ab1.iloc[-2][1]]  # actual u value [W/(m2K)]

A_door = [sfh1.iloc[-1][0], th1.iloc[-1][0], mfh1.iloc[-1][0], ab1.iloc[-1][0]]   # door envelope area [m2]
U_door = [sfh1.iloc[-1][1], th1.iloc[-1][15], mfh1.iloc[-1][15], ab1.iloc[-1][15]]   # actual u value [W/(m2K)]

A_env_sfh = np.array([A_wall1[0], A_wall2[0], A_roof[0], A_roof1[0],
                      A_floor[0], A_floor1[0], A_win[0], A_win1[0], A_door[0]]).sum()   # total envelope area [m2]
A_env_th = np.array([A_wall1[1], A_wall2[1], A_roof[1], A_roof1[1],
                      A_floor[1], A_floor1[1], A_win[1], A_win1[1], A_door[1]]).sum()   # total envelope area [m2]
A_env_mfh = np.array([A_wall1[2], A_wall2[2], A_roof[2], A_roof1[2],
                      A_floor[2], A_floor1[2], A_win[2], A_win1[2], A_door[2]]).sum()   # total envelope area [m2]
A_env_ab = np.array([A_wall1[3], A_wall2[3], A_roof[3], A_roof1[3],
                      A_floor[3], A_floor1[3], A_win[3], A_win1[3], A_door[3]]).sum()   # total envelope area [m2]

# TODO 2.11 Heat losses
# sfh = single family house
# th = two family house
# mfh = multi family house
# ab = Apartment block

Q_hloss_sfh = []         # Total heat loss [KWh/m^2.a]
Q_hloss_th = []          # Total heat loss [KWh/m^2.a]
Q_hloss_mfh = []         # Total heat loss [KWh/m^2.a]
Q_hloss_ab = []          # Total heat loss [KWh/m^2.a]

q_tr_coeff_sfh = []      # Transmission loss coeff sfh
q_tr_coeff_th = []       # Transmission loss coeff th
q_tr_coeff_mfh = []      # Transmission loss coeff mfh
q_tr_coeff_ab = []       # Transmission loss coeff ab

q_ve_coeff_sfh = []      # Ventilation loss coeff sfh
q_ve_coeff_th = []       # Ventilation loss coeff th
q_ve_coeff_mfh = []      # Ventilation loss coeff mfh
q_ve_coeff_ab = []       # Ventilation loss coeff ab
a=[]
for i in range(0, len(weather)):
    if weather["T2m"][i] <= base_temp:
        temp_diff_sfh = (weather["T2m"][i] - teta) * 0.001 * Fnu[0]
        temp_diff_th = (weather["T2m"][i] - teta) * 0.001 * Fnu[1]
        temp_diff_mfh = (weather["T2m"][i] - teta) * 0.001 * Fnu[2]
        temp_diff_ab = (weather["T2m"][i] - teta) * 0.001 * Fnu[3]
        a.append(temp_diff_sfh * -1)
    # ventilation loss SFH
        qve_loss_sfh = cp_air * (n_air_use + n_air_inf) * A_ref[0] * h_room
        q_ve_coeff_sfh.append(qve_loss_sfh)
    # ventilation loss TH
        qve_loss_th = cp_air * (n_air_use + n_air_inf) * A_ref[1] * h_room
        q_ve_coeff_th.append(qve_loss_th)
    # ventilation loss MFH
        qve_loss_mfh = cp_air * (n_air_use + n_air_inf) * A_ref[2] * h_room
        q_ve_coeff_mfh.append(qve_loss_mfh)
    # ventilation loss AB
        qve_loss_ab = cp_air * (n_air_use + n_air_inf) * A_ref[3] * h_room
        q_ve_coeff_ab.append(qve_loss_ab)
    # transmission loss SFH
        qtr_loss_coeff_sfh = (
                            (round(U_wall1[0] * A_wall1[0] * btr_ex) + round(U_wall2[0] * A_wall2[0] * btr_exw)
                            + round(U_roof[0] * A_roof[0] * btr_ex) + round(U_roof1[0] * A_roof1[0] * btr_ex)
                            + round(U_floor[0] * A_floor[0] * btr_floor) + round(U_floor1[0] * A_floor1[0] * btr_floor)
                            + round(U_win[0] * A_win[0] * btr_ex) + round(U_win1[0] * A_win1[0] * btr_ex)
                            + round(U_door[0] * A_door[0] * btr_ex) + round(DU_tbr * A_env_sfh * btr_ex))
                             )
        q_tr_coeff_sfh.append(qtr_loss_coeff_sfh)
        combine_loss = (qtr_loss_coeff_sfh + qve_loss_sfh) * temp_diff_sfh * -1
        Q_hloss_sfh.append(round(combine_loss))
    # transmission loss TH
        qtr_loss_coeff_th = (
                            (round(U_wall1[1] * A_wall1[1] * btr_ex) + round(U_wall2[1] * A_wall2[1] * btr_ex)
                            + round(U_roof[1] * A_roof[1] * btr_ex) + round(U_roof1[1] * A_roof1[1] * btr_ex)
                            + round(U_floor[1] * A_floor[1] * btr_floor) + round(U_floor1[1] * A_floor1[1] * btr_floor)
                            + round(U_win[1] * A_win[1] * btr_ex) + round(U_win1[1] * A_win1[1] * btr_ex)
                            + round(U_door[1] * A_door[1] * btr_ex) + round(DU_tbr * A_env_th * btr_ex))
                           )
        q_tr_coeff_th.append(qtr_loss_coeff_th)
        combine_loss1 = (qtr_loss_coeff_th + qve_loss_th) * temp_diff_th * -1
        Q_hloss_th.append(round(combine_loss1))
    # transmission loss MFH
        qtr_loss_coeff_mfh = (
                             (round(U_wall1[2] * A_wall1[2] * btr_ex) + round(U_wall2[2] * A_wall2[2] * btr_ex)
                             + round(U_roof[2] * A_roof[2] * btr_ex) + round(U_roof1[2] * A_roof1[2] * btr_ex)
                             + round(U_floor[2] * A_floor[2] * btr_floor) + round(U_floor1[2] * A_floor1[2] * btr_floor)
                             + round(U_win[2] * A_win[2] * btr_ex) + round(U_win1[2] * A_win1[2] * btr_ex)
                             + round(U_door[2] * A_door[2] * btr_ex) + round(DU_tbr * A_env_mfh * btr_ex))
                             )
        q_tr_coeff_mfh.append(qtr_loss_coeff_mfh)
        combine_loss2 = (qtr_loss_coeff_mfh + qve_loss_mfh) * temp_diff_mfh * -1
        Q_hloss_mfh.append(round(combine_loss2))
    # transmission loss AB
        qtr_loss_coeff_ab = (
                            (round((U_wall1[3] * A_wall1[3] * btr_ex)) + round((U_wall2[3] * A_wall2[3] * btr_ex))
                            + round((U_roof[3] * A_roof[3] * btr_ex)) + round((U_roof1[3] * A_roof1[3] * btr_ex))
                            + round((U_floor[3] * A_floor[3] * btr_floor)) + round((U_floor1[3] * A_floor1[3] * btr_floor))
                            + round((U_win[3] * A_win[3] * btr_ex)) + round((U_win1[3] * A_win1[3] * btr_ex))
                            + round((U_door[3] * A_door[3] * btr_ex)) + round((DU_tbr * A_env_ab * btr_ex)))
                            )
        q_tr_coeff_ab.append(qtr_loss_coeff_ab)
        combine_loss3 = (qtr_loss_coeff_ab + qve_loss_ab) * temp_diff_ab * -1
        Q_hloss_ab.append(combine_loss3)
    else:
        Q_hloss_sfh.append(0)
        q_tr_coeff_sfh.append(0)
        q_ve_coeff_sfh.append(0)

        Q_hloss_th.append(0)
        q_tr_coeff_th.append(0)
        q_ve_coeff_th.append(0)

        Q_hloss_mfh.append(0)
        q_tr_coeff_mfh.append(0)
        q_ve_coeff_mfh.append(0)

        Q_hloss_ab.append(0)
        q_tr_coeff_ab.append(0)
        q_ve_coeff_ab.append(0)
#print(sum(a))
coeff_tr_res_sfh = round(q_tr_coeff_sfh[0]) #/len(q_tr_coeff_sfh)
coeff_tr_res_th = round(q_tr_coeff_th[0]) #/len(q_tr_coeff_th)
coeff_tr_res_mfh = round(q_tr_coeff_mfh[0]) #/len(q_tr_coeff_mfh)
coeff_tr_res_ab = round(q_tr_coeff_ab[0]) #/len(q_tr_coeff_ab)

coeff_ve_res_sfh = round(q_ve_coeff_sfh[0]) #/ len(q_ve_coeff_sfh)
coeff_ve_res_th = round(q_ve_coeff_th[0]) #/len(q_ve_coeff_th)
coeff_ve_res_mfh = round(q_ve_coeff_mfh[0]) #/ #len(q_ve_coeff_mfh)
coeff_ve_res_ab = round(q_ve_coeff_ab[0]) #/len(q_ve_coeff_ab)

#print(coeff_tr_res_sfh, coeff_ve_res_sfh, coeff_ve_res_sfh, coeff_ve_res_th)
# TODO-6 Solar Heat Gains During Heating Season and Cooling season
Ff = 0.3         # Frame area fraction
Fw = 0.9         # non-perpendicular
ggl = 0.75       # solar energy transmittance
Int_gain = 3     # internal heat gains [W/m^2]

Fsh_hor = 0.8    # external shading in horizontal direction
Fsh_j = 0.6      # external shading in other dirn

Aw_hor = [0, 0, 0, 0]         # window area in east direction
Aw_est = [5.7, 0.0, 22.2, 323.1]     # window area in east direction
Aw_st = [6.3, 8.1, 243.2, 26.6]      # window area in south direction
Aw_wst = [8.9, 0.0, 22.2, 323.1]     # window area in east direction
Aw_nrt = [4.1, 5.4, 219.8, 14.3]     # window area in east direction

# TODO Heat Gain
Q_hgain_sfh = []
Qin_gain_sfh = []
Q_hgain_th = []
Qin_gain_th = []
Q_hgain_mfh = []
Qin_gain_mfh = []
Q_hgain_ab = []
Qin_gain_ab = []
for i in range(0, len(weather)):
    if weather["T2m"][i] <= base_temp:
        const = (1 - Ff) * Fw * ggl * 0.001
        const1 = (1 - Ff) * Fw * ggl * Fsh_j * 0.001
        q_gain_sfh= (
            (((Aw_est[0] * weather["G(i)_est"][i])    # gain in east direction
            + (Aw_st[0] * weather["G(i)_sth"][i])     # gain in south direction
            + (Aw_wst[0] * weather["G(i)_wst"][i])    # gain in west direction
            + (Aw_nrt[0] * weather["G(i)_nth"][i]))   # gain in north direction
            * const1) + (Fsh_hor * Aw_hor[0] * weather["G(i)_hor"][i] * const) # gain in horigontal direction
            )
        in_gain_sfh = Int_gain * A_ref[0] * 0.001   # Internal heat gain

        q_gain_th = (
                (((Aw_est[1] * weather["G(i)_est"][i])  # gain in east direction
                  + (Aw_st[1] * weather["G(i)_sth"][i])  # gain in south direction
                  + (Aw_wst[1] * weather["G(i)_wst"][i])  # gain in west direction
                  + (Aw_nrt[1] * weather["G(i)_nth"][i]))  # gain in north direction
                  * const1) + (Fsh_hor * Aw_hor[1] * weather["G(i)_hor"][i] * const)  # gain in horigontal direction
                   )
        in_gain_th = Int_gain * A_ref[1] * 0.001  # Internal heat gain

        q_gain_mfh = (
                (((Aw_est[2] * weather["G(i)_est"][i])  # gain in east direction
                  + (Aw_st[2] * weather["G(i)_sth"][i])  # gain in south direction
                  + (Aw_wst[2] * weather["G(i)_wst"][i])  # gain in west direction
                  + (Aw_nrt[2] * weather["G(i)_nth"][i]))  # gain in north direction
                 * const1) + (Fsh_hor * Aw_hor[2] * weather["G(i)_hor"][i] * const)  # gain in horigontal direction
                )
        in_gain_mfh = Int_gain * A_ref[2] * 0.001  # Internal heat gain

        q_gain_ab = (
                (((Aw_est[3] * weather["G(i)_est"][i])  # gain in east direction
                  + (Aw_st[3] * weather["G(i)_sth"][i])  # gain in south direction
                  + (Aw_wst[3] * weather["G(i)_wst"][i])  # gain in west direction
                  + (Aw_nrt[3] * weather["G(i)_nth"][i]))  # gain in north direction
                 * const1) + (Fsh_hor * Aw_hor[3] * weather["G(i)_hor"][i] * const)  # gain in horigontal direction

        )
        in_gain_ab = (Int_gain * A_ref[3] * 0.001) # Internal heat gain

        Q_hgain_sfh.append(round(q_gain_sfh))
        Qin_gain_sfh.append((in_gain_sfh))

        Q_hgain_th.append(round(q_gain_th))
        Qin_gain_th.append(in_gain_th)

        Q_hgain_mfh.append(round(q_gain_mfh))
        Qin_gain_mfh.append(in_gain_mfh)

        Q_hgain_ab.append(round(q_gain_ab))
        Qin_gain_ab.append(in_gain_ab)
    else:
        Q_hgain_sfh.append(0)
        Qin_gain_sfh.append(0)
        Q_hgain_th.append(0)
        Qin_gain_th.append(0)
        Q_hgain_mfh.append(0)
        Qin_gain_mfh.append(0)
        Q_hgain_ab.append(0)
        Qin_gain_ab.append(0)
#print(sum(Qin_gain_ab))

# TODO 8- Utilisation factor
# Utilisaton factor n_ff

cm = 45                       # internal heat capacity per square ref area
aH_o = 0.8                    # constant parameter for seasonal method [-]
TH_o = 30                     # contsant parameter [hours]

time_cons_sfh = round((cm * A_ref[0]) / (coeff_tr_res_sfh + coeff_ve_res_sfh))     # Building time constant
aH_sfh = aH_o + (time_cons_sfh / TH_o)
Yh_sfh = (sum(Q_hgain_sfh) + sum(Qin_gain_sfh)) / sum(Q_hloss_sfh)    # Heat balance ratio for the heating mode unscaled
n_ff_sfh = ((1 - (Yh_sfh ** aH_sfh)) / (1 - (Yh_sfh ** (aH_sfh + 1))))

time_cons_th = round((cm * A_ref[1]) / (coeff_tr_res_th + coeff_ve_res_th))     # Building time constant
aH_th = (aH_o + (time_cons_th / TH_o))
Yh_th = (sum(Q_hgain_th) + sum(Qin_gain_th)) / sum(Q_hloss_th)    # Heat balance ratio for the heating mode unscaled
n_ff_th = ((1 - (Yh_th ** aH_th)) / (1 - (Yh_th ** (aH_th + 1))))

time_cons_mfh = round((cm * A_ref[2]) / (coeff_tr_res_mfh + coeff_ve_res_mfh))     # Building time constant
aH_mfh = aH_o + (time_cons_mfh / TH_o)
Yh_mfh = (sum(Q_hgain_mfh) + sum(Qin_gain_mfh)) / sum(Q_hloss_mfh)   # Heat balance ratio for the heating mode unscaled
n_ff_mfh = ((1 - (Yh_mfh ** (aH_mfh))) / (1 - (Yh_mfh ** (aH_mfh + 1))))

time_cons_ab = round((cm * A_ref[3]) / (coeff_tr_res_ab + coeff_ve_res_ab))     # Building time constant
aH_ab = aH_o + (time_cons_ab / TH_o)
Yh_ab = (sum(Q_hgain_ab) + sum(Qin_gain_ab)) / sum(Q_hloss_ab)  # Heat balance ratio for the heating mode unscaled
n_ff_ab = (1 - (Yh_ab ** (aH_ab))) / (1 - (Yh_ab ** (aH_ab + 1)))


# TODO 10 - Energy Demand for heating with time constant
T_daily_avg = weather.resample('D').mean()
hourly_temp = pd.DataFrame(np.repeat(T_daily_avg['T2m'], 24), columns=['T2m'])

Qd_ht_sfh = []               # demand without time constant unscaled
Qd_ht_th = []
Qd_ht_mfh = []
Qd_ht_ab = []
for i in range(0, len(weather)):
    if hourly_temp["T2m"][i] > base_temp:
        Qd_ht_sfh.append(0)
        Qd_ht_th.append(0)
        Qd_ht_mfh.append(0)
        Qd_ht_ab.append(0)
    elif weather["T2m"][i] <= base_temp:
       Qd_ht_sfh.append(Q_hloss_sfh[i] - (n_ff_sfh * (Q_hgain_sfh[i] + Qin_gain_sfh[i])))  # Demand for heating with time constant
       Qd_ht_th.append(Q_hloss_th[i] - (n_ff_th * (Q_hgain_th[i] + Qin_gain_th[i])))
       Qd_ht_mfh.append(Q_hloss_mfh[i] - (n_ff_mfh * (Q_hgain_mfh[i] + Qin_gain_mfh[i])))
       Qd_ht_ab.append(Q_hloss_ab[i] - (n_ff_ab * (Q_hgain_ab[i] + Qin_gain_ab[i])))
    else:
        Qd_ht_sfh.append(0)
        Qd_ht_th.append(0)
        Qd_ht_mfh.append(0)
        Qd_ht_ab.append(0)

Q_demand_sfh = Qd_ht_sfh
Q_demand_th = Qd_ht_th
Q_demand_mfh = Qd_ht_mfh
Q_demand_ab = Qd_ht_ab
#print(len(Q_demand_sfh))
# TODO Ploting Condition tc
demand_sfh = []
demand_th = []
demand_mfh = []
demand_ab = []
for i in range(0, len(weather)):
    if Qd_ht_sfh[i] != 0:
        demand_sfh.append(Qd_ht_sfh[i])
    elif Qd_ht_sfh[i] == 0:
        demand_sfh.append(0)
    if Qd_ht_th[i] != 0:
        demand_th.append(Qd_ht_th[i])
    elif Qd_ht_th[i] == 0:
        demand_th.append(0)
    if Qd_ht_mfh[i] != 0:
        demand_mfh.append(Qd_ht_mfh[i])
    elif Qd_ht_mfh[i] == 0:
        demand_mfh.append(0)
    if Qd_ht_ab[i] != 0:
        demand_ab.append(Qd_ht_ab[i])
    elif Qd_ht_ab[i] == 0:
        demand_ab.append(0)

heating_demand68 = pd.DataFrame({'period': np.arange(len(weather)),
                              'sfh': demand_sfh,
                               'th': demand_th,
                               'mfh': demand_mfh,
                               'ab': demand_ab
                               }).set_index('period')

#heating_demand68.to_csv(("output/heat demand_1958-1968.csv"), index=False)


heatflow68 = pd.DataFrame({"Heat transmission": ["Heat loss", "Heat gain", "Heat demand"],
                              "SFH[KWh/a]": [round(sum(Q_hloss_sfh)), round(sum(Q_hgain_sfh)), round(sum(Q_demand_sfh)/A_ref[0])],
                              "TH[KWh/a]": [round(sum(Q_hloss_th)), round(sum(Q_hgain_th)), round(sum(Q_demand_th)/A_ref[1])],
                              "MFH[KWh/a]": [round(sum(Q_hloss_mfh)), round(sum(Q_hgain_mfh)), round(sum(Q_demand_mfh)/A_ref[2])],
                              "AB[KWh/a]": [round(sum(Q_hloss_ab)), round(sum(Q_hgain_ab)), round(sum(Q_demand_ab)/A_ref[3])],
                           }, index=None)
heatflow68.to_csv(("output/heatflow_1958-1968.csv"), index=False)


# Heat Demand Per Square Meter
total_demand68 = round(sum([sum(Q_demand_sfh), sum(Q_demand_th),   # total space heating demand for all building types kWh/a
                    sum(Q_demand_mfh), sum(Q_demand_ab)]))
total_demand68_sqm = round(total_demand68 / sum(A_ref),0)
heatflow68_sqm = pd.DataFrame({'Heatiing demand [kWh/m^2.a]':[heatflow68["SFH[KWh/a]"][2],heatflow68["TH[KWh/a]"][2],
                                       heatflow68["MFH[KWh/a]"][2],heatflow68["AB[KWh/a]"][2], total_demand68_sqm],
                              'Building types': ['SFH', 'TH', 'MFH', 'AB', 'Total Demand']}, index=None)

a_x = sns.barplot(x='Building types',y = 'Heatiing demand [kWh/m^2.a]',
                  width=0.25,data=heatflow68_sqm, color='midnightblue', label='from 1958-1968')
for i in a_x.containers:
    a_x.bar_label(i,)
plt.xlabel('Building types', fontsize=18)
plt.ylabel('Heating demand [KWh/m^2.a]', fontsize=18)
plt.yticks([0, 50, 100, 150, 200, 250],fontsize=18)
plt.xticks([r for r in range(len(heatflow68_sqm))],
        ['SFH', 'TH', 'MFH', 'AB', 'Total Demand'])
plt.title('Space heating demand per square meter', fontsize=18)
plt.legend()
plt.ticklabel_format()
plt.tight_layout()
plt.savefig(f'output/Demand_sqm from 1958-1968')
plt.show()


#fig = plt.rcParams["figure.figsize"] = (10,6)
plt.stackplot(heating_demand68.index, heating_demand68["sfh"], heating_demand68["th"], heating_demand68["mfh"],
              heating_demand68["ab"], colors=['gold', 'limegreen', 'purple', 'midnightblue'], labels=['SFH', 'TH', 'MFH', 'AB'])
plt.ylim(0, None)
plt.xlim(0, len(demand_sfh))
plt.xlabel("Time Period (h)", fontsize=18)
plt.title(f'Space Heating Demand Profile', fontsize=18)
plt.ylabel('Power (KW)', fontsize=18)
plt.yticks([0, 100, 200, 300, 400, 500],fontsize=18)
plt.xticks([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016],
      ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'], fontsize=16)
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.ticklabel_format()
plt.savefig(f'output/1958-1968')
plt.show()

# TODO Yearly Duration Curve
heating_demand['Heatlost'] = heating_demand[list(heating_demand.columns)].sum(axis=1) # Add the two columns together
print(heating_demand)
heating_demand_array = heating_demand['Heatlost'].values  # sorting values from large to small
print(heating_demand_array)
hourly_array = np.arange(len(weather), dtype=int)         # lenght of heating period
print(hourly_array)
Lost_data = {'hours': hourly_array, 'Heatlost': heating_demand_array}
Lost_df = pd.DataFrame(Lost_data)                         # create a datafram of sorted values
print(Lost_df)

sns.set(rc={"figure.figsize":(10, 6)})
p = sns.lineplot(x = "hours", y = "Heatlost", data = Lost_df)
plt.ylim(0, None)
plt.xlim(0, len(weather))
p.set_title(f"Load duration curve for MFH", fontsize = 18)
p.set_xlabel("Time (hours)", fontsize = 18)
p.set_ylabel("Power (KW)", fontsize = 18)
plt.xticks([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016],
      ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'], fontsize=16)
plt.ticklabel_format()
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig(f'output/year load profile 1958-1968')
plt.show()

Lost_df['interval'] = 1
Lost_df_sorted = Lost_df.sort_values(by=['Heatlost'], ascending=False)
Lost_df_sorted['Duration'] = Lost_df_sorted['interval'].cumsum()
Lost_df_sorted['Percentage'] = Lost_df_sorted['Duration']*100/len(weather)

p = sns.lineplot(x = "Duration", y = "Heatlost", data = Lost_df_sorted)
plt.ylim(0, None)
plt.xlim(0, len(weather))
p.set_title(f"Load duration curve for MFH", fontsize = 16)
p.set_xlabel("Time (hours)", fontsize = 16)
p.set_ylabel("Power (KW)", fontsize = 16)
plt.xticks([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016],
      ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])
plt.ticklabel_format()
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig(f'output/LDC_HL_1958-1968')
plt.show()
