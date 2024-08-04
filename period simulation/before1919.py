# TODO-1 Importing Tabula Parameter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
#rcParams['font.weight'] = 'bold'

sfh = "C:/Users/elvis/Desktop/Pyclass/Thesis/Thesis_data_simulation/parameter_SFH.csv"
th = "C:/Users/elvis/Desktop/Pyclass/Thesis/Thesis_data_simulation/parameter_TH.csv"
mfh = "C:/Users/elvis/Desktop/Pyclass/Thesis/Thesis_data_simulation/parameter_MFH.csv"
ab = "C:/Users/elvis/Desktop/Pyclass/Thesis/Thesis_data_simulation/parameter_AB.csv"
sfh1 = pd.read_csv(sfh,header=0,nrows=10, index_col=0)
th1 = pd.read_csv(th,header=0,nrows=10, index_col=0)
mfh1 = pd.read_csv(mfh,header=0,nrows=10, index_col=0)
ab1 = pd.read_csv(ab,nrows=10, index_col=0)

# TODO - Session_2 : Weather Data
simulation_data = "C:/Users/elvis/Desktop/Heating Demand/working data.csv"
weather = pd.read_csv(simulation_data, skiprows=8, nrows=8760, index_col=0)
weather.index = pd.to_datetime(weather.index, format="%Y%m%d:%H%M")
weather.index = pd.date_range(start="2009-01-01 00:00", end="2009-12-31 23:00", freq="h")
"""
temp= weather["T2m"]
a = temp #.resample("h").mean()
plt.plot(a)
plt.title(f"External temperature profile", fontsize =14)
plt.xlabel("Reference Year", fontsize =14)
plt.ylabel("Temperature values (°C)", fontsize =14)
plt.yticks([-20, -10, 0, 10, 20, 30],fontsize=12)
plt.xticks(["2009-01", "2009-03" , "2009-05", "2009-07", "2009-09", "2009-11", "2010-01"],
           ['Jan', 'Mar', 'Mai', 'Jul', 'Sep', 'Nov', 'Jan'])
plt.ticklabel_format()
plt.tight_layout()
plt.savefig(f'output/before 1919_temp')
plt.show()"""
# TODO - 2.1 Hourly Calaculation

# Tabula Data
teta = 20.0           # indoor temp heating season
A_ref = [141.8, 96, 312.4, 829.4]   # reference area for sfh, th, mfh and ab in [m2]
btr_ex = 1.0          # adjustment soil factor for construction bordering on external air
btr_floor = 0.5       # adjustment soil factor of unheated cellar/construction bordering on soil
Fnu = [0.80, 0.84, 0.90, 0.93]    # temp reduction sfh, th, mfh and ab (high thermal conductivity
base_temp = 12.0      # Base temp [°C]
DU_tbr = 0.10         # U value for thermal bridge

cp_air = 0.34              # Volume specific heat capacity of air            [Wh/(m^3.K)]
n_air_use = 0.40           # Air change rate by use                          [1/h]
n_air_inf = 0.20           # Air change rate by infiltration                 [1/h]
h_room = 2.5               # standard value of room height                   [m]

A_wall1 = [sfh1.iloc[2][0], th1.iloc[2][0], mfh1.iloc[2][0], ab1.iloc[2][0]]     # Wall envelope area [m2]
U_wall1 = [sfh1.iloc[2][1], th1.loc["wall_1", "Uactual,iW/(m2K)"],
          mfh1.loc["wall_1", "Uactual,iW/(m2K)"], ab1.iloc[2][15]]     # Heat transfer coeff - actual u value [W/(m2K)]
A_wall2 = [sfh1.iloc[3][0], th1.iloc[3][0], mfh1.iloc[3][0], ab1.iloc[3][0]]     # Wall envelope area [m2]
U_wall2 = [sfh1.iloc[3][1], th1.loc["wall_2", "Uactual,iW/(m2K)"],
           mfh1.loc["wall_2", "Uactual,iW/(m2K)"], ab1.iloc[3][15]]    # Heat transfer coeff - actual u value [W/(m2K)]
U_roof = [sfh1.loc["roof_1", "Uactual,iW/(m2K)"],
          th1.loc["roof_1", "Uactual,iW/(m2K)"],
          mfh1.loc["roof_1", "Uactual,iW/(m2K)"],
          ab1.loc["roof_1", "Uactual,iW/(m2K)"]]     # actual u value [W/(m2K)]
A_roof = [sfh1.iloc[0][0], th1.iloc[0][0], mfh1.iloc[0][0], ab1.iloc[0][0]]                       # roof envelope area [m2]
U_roof1 = [sfh1.loc["roof_2", "Uactual,iW/(m2K)"], th1.loc["roof_2", "Uactual,iW/(m2K)"],
           mfh1.loc["roof_2", "Uactual,iW/(m2K)"], ab1.iloc[1][15]]   # actual u value [W/(m2K)]
A_roof1 = [sfh1.iloc[1][0], th1.iloc[1][0], mfh1.iloc[1][0], ab1.iloc[1][0]]                         # roof envelope area [m2]
U_floor = [sfh1.loc["floor_2", "Uactual,iW/(m2K)"], th1.loc["floor_1", "Uactual,iW/(m2K)"],
           mfh1.loc["floor_1", "Uactual,iW/(m2K)"], ab1.iloc[5][15]] # actual u value [W/(m2K)]
A_floor = [sfh1.iloc[5][0], th1.iloc[5][0], mfh1.iloc[5][0], ab1.iloc[5][0]]                        # floor envelope area [m2]
U_floor1 = [sfh1.iloc[5][1], th1.loc["floor_2", "Uactual,iW/(m2K)"],
            mfh1.loc["floor_2", "Uactual,iW/(m2K)"], ab1.iloc[6][15]]                         # actual u value [W/(m2K)]
A_floor1 = [sfh1.iloc[-4][0], th1.iloc[-4][0], mfh1.iloc[-4][0], ab1.iloc[-4][0]]                       # floor envelope area [m2]
U_win = [sfh1.loc["window_1", "Uactual,iW/(m2K)"], th1.loc["window_1", "Uactual,iW/(m2K)"],
         mfh1.loc["window_1", "Uactual,iW/(m2K)"], ab1.iloc[-3][15]]  # actual u value [W/(m2K)]
A_win = [sfh1.iloc[-3][0], th1.iloc[-3][0], mfh1.iloc[-3][0], ab1.iloc[-3][0]]                          # window envelope area [m2]
U_win1 = [sfh1.loc["window_2", "Uactual,iW/(m2K)"], th1.loc["window_2", "Uactual,iW/(m2K)"],
          mfh1.loc["window_2", "Uactual,iW/(m2K)"], ab1.iloc[-2][15]]  # actual u value [W/(m2K)]
A_win1 = [sfh1.iloc[-2][0], th1.iloc[-2][0], mfh1.iloc[-2][0], ab1.iloc[-2][0]]                         # window envelope area [m2]
U_door = [sfh1.loc["door_1", "Uactual,iW/(m2K)"], th1.loc["door_1", "Uactual,iW/(m2K)"],
          mfh1.loc["door_1", "Uactual,iW/(m2K)"], ab1.loc["door_1", "Uactual,iW/(m2K)"]]  # actual u value [W/(m2K)]
A_door = [sfh1.iloc[-1][0], th1.iloc[-1][0], mfh1.iloc[-1][0], ab1.iloc[-1][0]]                        # door envelope area [m2]

A_env_sfh = np.array([A_wall1[0], A_wall2[0], A_roof[0], A_roof1[0],
                      A_floor[0], A_floor1[0], A_win[0], A_win1[0], A_door[0]]).sum()   # total envelope area [m2]
A_env_th = np.array([A_wall1[1], A_wall2[1], A_roof[1], A_roof1[1],
                      A_floor[1], A_floor1[1], A_win[1], A_win1[1], A_door[1]]).sum()   # total envelope area [m2]
A_env_mfh = np.array([A_wall1[2], A_wall2[2], A_roof[2], A_roof1[2],
                      A_floor[2], A_floor1[2], A_win[2], A_win1[2], A_door[2]]).sum()   # total envelope area [m2]
A_env_ab = np.array([A_wall1[3], A_wall2[3], A_roof[3], A_roof1[3],
                      A_floor[3], A_floor1[3], A_win[3], A_win1[3], A_door[3]]).sum()   # total envelope area [m2]


# TODO 2.11 Heat losses

Q_hloss_sfh = []          # Total heat loss [KWh/m^2.a]
Q_hloss_th = []          # Total heat loss [KWh/m^2.a]
Q_hloss_mfh = []
Q_hloss_ab = []

q_tr_coeff_sfh = []   # Transmission loss coeff sfh
q_tr_coeff_th = []   # Transmission loss coeff th
q_tr_coeff_mfh = []
q_tr_coeff_ab = []

q_ve_coeff_sfh = []
q_ve_coeff_th = []
q_ve_coeff_mfh = []
q_ve_coeff_ab = []
for i in range(0, len(weather)):
    if weather["T2m"][i] <= base_temp:
        temp_diff_sfh = (weather["T2m"][i] - teta) * 0.001 * Fnu[0]
        temp_diff_th = (weather["T2m"][i] - teta) * 0.001 * Fnu[1]
        temp_diff_mfh = (weather["T2m"][i] - teta) * 0.001 * Fnu[2]
        temp_diff_ab = (weather["T2m"][i] - teta) * 0.001 * Fnu[3]

    # ventilation loss SFH
        qve_loss_sfh = (cp_air * (n_air_use + n_air_inf) * A_ref[0] * h_room) * (temp_diff_sfh / Fnu[0])
        Qve_coeff_sfh = cp_air * (n_air_use + n_air_inf) * A_ref[0] * h_room
        q_ve_coeff_sfh.append(Qve_coeff_sfh)
    # ventilation loss TH
        qve_loss_th = (cp_air * (n_air_use + n_air_inf) * A_ref[1] * h_room) * (temp_diff_th / Fnu[1])
        Qve_coeff_th = cp_air * (n_air_use + n_air_inf) * A_ref[1] * h_room
        q_ve_coeff_th.append(Qve_coeff_th)
    # ventilation loss MFH
        qve_loss_mfh = cp_air * (n_air_use + n_air_inf) * A_ref[2] * h_room * (temp_diff_mfh / Fnu[2])
        Qve_coeff_mfh = cp_air * (n_air_use + n_air_inf) * A_ref[2] * h_room
        q_ve_coeff_mfh.append(Qve_coeff_mfh)
    # ventilation loss AB
        qve_loss_ab = (cp_air * (n_air_use + n_air_inf) * A_ref[3] * h_room) * (temp_diff_th / Fnu[3])
        Qve_coeff_ab = cp_air * (n_air_use + n_air_inf) * A_ref[3] * h_room
        q_ve_coeff_ab.append(Qve_coeff_ab)
    # transmission loss SFH
        qtr_loss_coeff_sfh = (
           ((U_wall1[0] * A_wall1[0] * btr_ex) + (U_wall2[0] * A_wall2[0] * btr_ex)
             + (U_roof[0] * A_roof[0] * btr_ex) + (U_roof1[0] * A_roof1[0] * btr_ex)
             + (U_floor[0] * A_floor[0] * btr_floor) + (U_floor1[0] * A_floor1[0] * btr_floor)
             + (U_win[0] * A_win[0] * btr_ex) + (U_win1[0] * A_win1[0] * btr_ex)
             + (U_door[0] * A_door[0] * btr_ex) + (DU_tbr * A_env_sfh * btr_ex))
               )
        q_tr_coeff_sfh.append(qtr_loss_coeff_sfh)
        qtr_loss_sfh = qtr_loss_coeff_sfh * temp_diff_sfh
        combine_loss = (-1 * qtr_loss_sfh) + (-1 * qve_loss_sfh)
        Q_hloss_sfh.append(combine_loss)
    # transmission loss TH
        qtr_loss_coeff_th = (
                ((U_wall1[1] * A_wall1[1] * btr_ex) + (U_wall2[1] * A_wall2[1] * btr_ex)
                 + (U_roof[1] * A_roof[1] * btr_ex) + (U_roof1[1] * A_roof1[1] * btr_ex)
                 + (U_floor[1] * A_floor[1] * btr_floor) + (U_floor1[1] * A_floor1[1] * btr_floor)
                 + (U_win[1] * A_win[1] * btr_ex) + (U_win1[1] * A_win1[1] * btr_ex)
                 + (U_door[1] * A_door[1] * btr_ex) + (DU_tbr * A_env_th * btr_ex))
        )
        q_tr_coeff_th.append(qtr_loss_coeff_th)
        qtr_loss_th = qtr_loss_coeff_th * temp_diff_th
        combine_loss1 = (-1 * qtr_loss_th) + (-1 * qve_loss_th)
        Q_hloss_th.append(combine_loss1)
    # transmission loss MFH
        qtr_loss_coeff_mfh = (
            ((U_wall1[2] * A_wall1[2] * btr_ex) + (U_wall2[2] * A_wall2[2] * btr_ex)
             + (U_roof[2] * A_roof[2] * btr_ex) + (U_roof1[2] * A_roof1[2] * btr_ex)
             + (U_floor[2] * A_floor[2] * btr_floor) + (U_floor1[2] * A_floor1[2] * btr_floor)
             + (U_win[2] * A_win[2] * btr_ex) + (U_win1[2] * A_win1[2] * btr_ex)
             + (U_door[2] * A_door[2] * btr_ex) + (DU_tbr * A_env_mfh * btr_ex))
        )
        q_tr_coeff_mfh.append(qtr_loss_coeff_mfh)
        qtr_loss_mfh = round(qtr_loss_coeff_mfh) * temp_diff_mfh
        combine_loss2 = (-1 * qtr_loss_mfh) + (-1 * qve_loss_mfh)
        Q_hloss_mfh.append(combine_loss2)
    # transmission loss AB
        qtr_loss_coeff_ab = (
            (round((U_wall1[3] * A_wall1[3] * btr_ex)) + round((U_wall2[3] * A_wall2[3] * btr_ex))
             + round((U_roof[3] * A_roof[3] * btr_ex)) + round((U_roof1[3] * A_roof1[3] * btr_ex))
             + round((U_floor[3] * A_floor[3] * btr_floor)) + round((U_floor1[3] * A_floor1[3] * btr_floor))
             + round((U_win[3] * A_win[3] * btr_ex)) + round((U_win1[3] * A_win1[3] * btr_ex))
             + round((U_door[3] * A_door[3] * btr_ex)) + round((DU_tbr * A_env_ab * btr_ex)))
        )
        q_tr_coeff_ab.append(qtr_loss_coeff_ab)
        qtr_loss_ab = qtr_loss_coeff_ab * temp_diff_ab
        combine_loss3 = (-1 * qtr_loss_ab) + (-1 * qve_loss_ab)
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
coeff_tr_res_sfh = sum(q_tr_coeff_sfh)/len(q_tr_coeff_sfh)
coeff_tr_res_th = sum(q_tr_coeff_th)/len(q_tr_coeff_th)
coeff_tr_res_mfh = sum(q_tr_coeff_mfh)/len(q_tr_coeff_mfh)
coeff_tr_res_ab = sum(q_tr_coeff_ab)/len(q_tr_coeff_ab)

coeff_ve_res_sfh = sum(q_ve_coeff_sfh)/len(q_ve_coeff_sfh)
coeff_ve_res_th = sum(q_ve_coeff_th)/len(q_ve_coeff_th)
coeff_ve_res_mfh = sum(q_ve_coeff_mfh)/len(q_ve_coeff_mfh)
coeff_ve_res_ab = sum(q_ve_coeff_ab)/len(q_ve_coeff_ab)

#print(coeff_tr_res_sfh, coeff_tr_res_th, coeff_ve_res_sfh, coeff_ve_res_th)
# TODO-6 Solar Heat Gains During Heating Season and Cooling season
Ff = 0.3         # Frame area fraction
Fw = 0.9         # non-perpendicular
ggl = 0.75       # solar energy transmittance
Int_gain = 3.0   # internal heat gains [W/m^2]

Fsh_hor = 0.8    # external shading in horizontal direction
Fsh_j = 0.6      # external shading in other dirn

Aw_hor = [0, 0, 0, 0]         # window area in east direction
Aw_est = [7.7, 0.0, 26.4, 65.8]     # window area in east direction
Aw_st = [5.6, 8.0, 0.0, 2.3]      # window area in south direction
Aw_wst = [7.7, 0.0, 26.4, 65.8]     # window area in east direction
Aw_nrt = [1.4, 10.1, 1.3, 2.3]     # window area in east direction

# TODO Heat Gain
Q_hgain_sfh = []
Q_hgain_th = []
Q_hgain_mfh = []
Q_hgain_ab = []
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
            + (Int_gain * A_ref[0] * 0.001 )  # Internal heat gain
               )

        q_gain_th = (
                (((Aw_est[1] * weather["G(i)_est"][i])  # gain in east direction
                  + (Aw_st[1] * weather["G(i)_sth"][i])  # gain in south direction
                  + (Aw_wst[1] * weather["G(i)_wst"][i])  # gain in west direction
                  + (Aw_nrt[1] * weather["G(i)_nth"][i]))  # gain in north direction
                 * const1) + (Fsh_hor * Aw_hor[1] * weather["G(i)_hor"][i] * const)  # gain in horigontal direction
                + (Int_gain * A_ref[1] * 0.001)  # Internal heat gain
        )

        q_gain_mfh = (
                (((Aw_est[2] * weather["G(i)_est"][i])  # gain in east direction
                  + (Aw_st[2] * weather["G(i)_sth"][i])  # gain in south direction
                  + (Aw_wst[2] * weather["G(i)_wst"][i])  # gain in west direction
                  + (Aw_nrt[2] * weather["G(i)_nth"][i]))  # gain in north direction
                 * const1) + (Fsh_hor * Aw_hor[2] * weather["G(i)_hor"][i] * const)  # gain in horigontal direction
                + (Int_gain * A_ref[2] * 0.001)  # Internal heat gain
        )
        q_gain_ab = (
                (((Aw_est[3] * weather["G(i)_est"][i])  # gain in east direction
                  + (Aw_st[3] * weather["G(i)_sth"][i])  # gain in south direction
                  + (Aw_wst[3] * weather["G(i)_wst"][i])  # gain in west direction
                  + (Aw_nrt[3] * weather["G(i)_nth"][i]))  # gain in north direction
                 * const1) + (Fsh_hor * Aw_hor[3] * weather["G(i)_hor"][i] * const)  # gain in horigontal direction
                + (Int_gain * A_ref[3] * 0.001)  # Internal heat gain
        )

        Q_hgain_sfh.append(round(q_gain_sfh))
        Q_hgain_th.append(round(q_gain_th))
        Q_hgain_mfh.append(round(q_gain_mfh))
        Q_hgain_ab.append(round(q_gain_ab))
    else:
        Q_hgain_sfh.append(0)
        Q_hgain_th.append(0)
        Q_hgain_mfh.append(0)
        Q_hgain_ab.append(0)
#print(len(Q_hgain))

# TODO 8- Utilisation factor
# Utilisaton factor n_ff

cm = 45                       # internal heat capacity per square ref area
aH_o = 0.8                    # constant parameter for seasonal method [-]
TH_o = 30                     # contsant parameter [hours]

time_cons_sfh = round(cm * A_ref[0] / (coeff_tr_res_sfh +coeff_ve_res_sfh))     # Building time constant
aH_sfh = round(aH_o + time_cons_sfh / TH_o,2)
Yh_sfh = round(sum(Q_hgain_sfh) / sum(Q_hloss_sfh),3)    # Heat balance ratio for the heating mode unscaled
n_ff_sfh = round((1 - (Yh_sfh ** (aH_sfh))) / (1 - (Yh_sfh ** (aH_sfh + 1))),2)

time_cons_th = round(cm * A_ref[1] / (coeff_tr_res_th + coeff_ve_res_th))     # Building time constant
aH_th = round(aH_o + time_cons_th / TH_o ,2)
Yh_th = round(sum(Q_hgain_th) / sum(Q_hloss_th),3)    # Heat balance ratio for the heating mode unscaled
n_ff_th = round((1 - (Yh_th ** (aH_th))) / (1 - (Yh_th ** (aH_th + 1))),2)

time_cons_mfh = round(cm * A_ref[2] / (coeff_tr_res_mfh + coeff_ve_res_mfh))     # Building time constant
aH_mfh = round(aH_o + time_cons_mfh / TH_o,3)
Yh_mfh = round(sum(Q_hgain_mfh) / sum(Q_hloss_mfh),3)   # Heat balance ratio for the heating mode unscaled
n_ff_mfh = round((1 - (Yh_mfh ** (aH_mfh))) / (1 - (Yh_mfh ** (aH_mfh + 1))),2)

time_cons_ab = round(cm * A_ref[3] / (coeff_tr_res_ab + coeff_ve_res_ab))     # Building time constant
aH_ab = round(aH_o + time_cons_ab / TH_o ,2)
Yh_ab = round(sum(Q_hgain_ab) / sum(Q_hloss_ab),3)    # Heat balance ratio for the heating mode unscaled
n_ff_ab = round((1 - (Yh_ab ** (aH_ab))) / (1 - (Yh_ab ** (aH_ab + 1))),2)


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
       Qd_ht_sfh.append((Q_hloss_sfh[i] - n_ff_sfh * Q_hgain_sfh[i]))  # Demand for heating with time constant
       Qd_ht_th.append((Q_hloss_th[i] - n_ff_th * Q_hgain_th[i]))
       Qd_ht_mfh.append((Q_hloss_mfh[i] - n_ff_mfh * Q_hgain_mfh[i]))
       Qd_ht_ab.append((Q_hloss_ab[i] - n_ff_ab * Q_hgain_ab[i]))
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

heating_demand19 = pd.DataFrame({'period': np.arange(len(weather)),
                              'sfh': demand_sfh,
                              'th': demand_th,
                              'mfh': demand_mfh,
                              'ab': demand_ab
                               }).set_index('period')


heating_demand19.to_csv(("output/heat demand before 1919.csv"), index=False)


heatflow = pd.DataFrame({"Heat transmission": ["Heat loss", "Heat gain", "Heat demand", "flow per sqm"],
                              "SFH[KWh/a]": [round(sum(Q_hloss_sfh)), round(sum(Q_hgain_sfh)),
                                             round(sum(Q_demand_sfh)), round(sum(Q_demand_sfh) / A_ref[0])],
                              "TH[KWh/a]": [round(sum(Q_hloss_th)), round(sum(Q_hgain_th)),
                                            round(sum(Q_demand_th)), round(sum(Q_demand_th) / A_ref[1])],
                              "MFH[KWh/a]": [round(sum(Q_hloss_mfh)), round(sum(Q_hgain_mfh)),
                                             round(sum(Q_demand_mfh)), round(sum(Q_demand_mfh) / A_ref[2])],
                              "AB[KWh/a]": [round(sum(Q_hloss_ab)), round(sum(Q_hgain_ab)),
                                            round(sum(Q_demand_ab)), round(sum(Q_demand_ab) / A_ref[3])],}, index=None)

heatflow.to_csv(("output/heatflow before 1919.csv"), index=False)

""""
flow_sqm = pd.DataFrame({"Power_KWh_sqm_year": [heatflow.iloc[-1][1], heatflow.iloc[-1][2], heatflow.iloc[-1][3],  heatflow.iloc[-1][4]],
                       "Building Type":  ["SFH", "TH", "MFH", "AB"]}, index=None)

print(flow_sqm)
cols = ['darkgreen' if (x == 271) else 'darkred' for x in flow_sqm.Power_KWh_sqm_year]
#plt.rcParams["figure.figsize"] = (8,6)
a_x = sns.barplot(x = 'Building Type',y = 'Power_KWh_sqm_year',
                  width=0.25,data = flow_sqm, palette=cols)
for i in a_x.containers:
    a_x.bar_label(i,)
plt.xlabel("Building Types", fontsize=12)
plt.ylabel("Power_kWh_sqm_year",fontsize = 12)
plt.yticks([0, 50, 100, 150, 200, 250], fontsize=12)
plt.title("Annual heat flow related to reference area",fontsize = 12)
plt.ticklabel_format()
plt.tight_layout()
plt.savefig(f'output/flow_sqm before 1919')
plt.show()


Building_Type = flow_sqm["Power_kWh_sqm_year"]
values = flow_sqm["Building Type"]

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(values, Building_Type, color =cols, width=0.4)

plt.xlabel("Building Types", weight='bold')
plt.ylabel("Power_KWh_sqm_year", weight='bold')
plt.title("Annual heat Flow Related To Reference Area", weight='bold')
plt.ticklabel_format()
plt.tight_layout()
#plt.savefig(f'output/flow_sqm before 1919')
plt.show()

fig, ax = plt.subplots()
labels=['AB']#, 'TH', 'MFH', 'AB']
ax.stackplot(heating_demand19.index, heating_demand19['ab'],
             colors=['green'], labels=labels)


plt.ylim(0, None)
plt.xlim(0, len(weather))
plt.xlabel("Time period (h)", fontsize = 14)
plt.title(f'Space heating demand profile',fontsize = 14)
plt.ylabel('Power (kW)', fontsize = 14)
plt.yticks([0, 10, 20, 30, 40, 50],fontsize=14)
plt.xticks([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016],
      ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'],fontsize=14)
plt.ticklabel_format()
plt.tight_layout()
plt.legend(loc="upper right")
#plt.grid()
plt.savefig(f'output/before 1919_AB')
plt.show()


# TODO Yearly Duration Curve
heating_demand19['Heatlost'] = heating_demand19[list(heating_demand19.columns)].sum(axis=1) # Add the two columns together

heating_demand_array = heating_demand19['Heatlost'].values  # sorting values from large to small
hourly_array = np.arange(len(weather), dtype=int)           # lenght of heating period

Lost_data = {'hours': hourly_array, 'Heatlost': heating_demand_array}
Lost_df = pd.DataFrame(Lost_data)    # create a datafram of sorted values
print(Lost_df)
sns.set(rc={"figure.figsize":(10, 6)})
p = sns.lineplot(x = "hours", y = "Heatlost", data = Lost_df)
plt.ylim(0, None)
plt.xlim(0, len(weather))
p.set_title(f"Load duration curve for SFH", fontsize = 14)
p.set_xlabel("Time (hours)", fontsize = 14,)
p.set_ylabel("Power (KW)", fontsize =14)
plt.xticks([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016],
      ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])
plt.legend()
plt.ticklabel_format()
plt.tight_layout()
plt.grid(color='red')
plt.tight_layout()
plt.savefig(f'output/year load profile before 1919')
plt.show()



Lost_df['interval'] = 1
Lost_df_sorted = Lost_df.sort_values(by=['Heatlost'], ascending=False)
Lost_df_sorted['Duration'] = Lost_df_sorted['interval'].cumsum()
Lost_df_sorted['Percentage'] = Lost_df_sorted['Duration']*100/8760

p = sns.lineplot(x = "Duration", y = "Heatlost", data = Lost_df_sorted)
plt.ylim(0, None)
plt.xlim(0, len(weather))
p.set_title(f"Load duration curve for MFH", fontsize=18)
p.set_xlabel("Time (hours)", fontsize =18)
p.set_ylabel("Power (kW)", fontsize=18)
plt.yticks([0, 5, 10, 15, 20, 25],fontsize=18)
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],fontsize=18)
 #     ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])
plt.ticklabel_format()
plt.tight_layout()
plt.grid(color='red')
plt.savefig(f'output/LDC_HL_before 1919_AB')
plt.show()

"""