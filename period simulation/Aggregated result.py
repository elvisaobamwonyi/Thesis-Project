import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from before1919  import heating_demand19
from btw1919_1948  import heating_demand48
from btw1949_1957  import heating_demand57
from btw1958_1968  import heating_demand68
from btw1969_1978  import heating_demand78
from btw1979_1983  import heating_demand83
from btw1984_1994  import heating_demand94
from btw1995_2001  import heating_demand01
from btw2002_2009  import heating_demand09
from btw2010_2015  import heating_demand15
from from2016 import heating_demand16
from matplotlib import rcParams
#rcParams['font.weight'] = 'bold'
building = pd.read_csv("C:/Users/elvis/Desktop/no. of Building in Thuringia.csv")
print(building)
#print(heating_demand19)
A_ref = [141.8, 302.5, 111.1, 121.2, 173.2, 215.6, 150.2,121.9,146.5, 186.8, 186.8]
#print((A_ref[0] * building.iloc[0][2]))

# Space heating demand for SFH according to construction period per sqm

res19 = (building.iloc[0][2] * heating_demand19["sfh"].sum()) / (A_ref[0] * building.iloc[0][2]) # demand 4 sfh b4 1919
res48 = (building.iloc[1][2] * heating_demand48["sfh"].sum()) / (A_ref[1] * building.iloc[1][2])  # demand 4 sfh b4 1948
res57 = (building.iloc[2][2] * heating_demand57["sfh"].sum()) / (A_ref[2] * building.iloc[2][2])  # demand 4 sfh b4 1957
res68 = (building.iloc[3][2] * heating_demand68["sfh"].sum()) / (A_ref[3] * building.iloc[3][2])  # demand 4 sfh b4 1968
res78 = (building.iloc[4][2] * heating_demand78["sfh"].sum()) / (A_ref[4] * building.iloc[4][2])  # demand 4 sfh b4 1978
res83 = (building.iloc[5][2] * heating_demand83["sfh"].sum()) / (A_ref[5] * building.iloc[5][2])  # demand 4 sfh b4 1983
res94 = (building.iloc[6][2] * heating_demand94["sfh"].sum()) / (A_ref[6] * building.iloc[6][2])  # demand 4 sfh b4 1994
res01 = (building.iloc[7][2] * heating_demand01["sfh"].sum()) / (A_ref[7] * building.iloc[7][2])  # demand 4 sfh b4 2001
res09 = (building.iloc[8][2] * heating_demand09["sfh"].sum()) / (A_ref[8] * building.iloc[8][2]) # demand 4 sfh b4 2009
res15 = (building.iloc[9][2] * heating_demand15["sfh"].sum()) / (A_ref[9] * building.iloc[9][2])  # demand 4 sfh b4 2015
res16 = (building.iloc[10][2] * heating_demand16["sfh"].sum()) / (A_ref[10] * building.iloc[10][2])  # demand 4 sfh from 2016
#print(building.iloc[5][2])

res_sfh = pd.DataFrame({ "construction_period": ["before 1919", "1919-1948", "1949-1957", "1958-1968", "1969-1978",
                                                 "1979-1983", "1984-1994", "1995-2001", "2002-2009", "2010-2015", "from 2016"],
                        "res19":[round(res19,2), round(res48,2), round(res57,2),
                                 round(res68,2), round(res78,2), round(res83,2), round(res94,2),
                                 round(res01,2), round(res09,2), round(res15,2), round(res16,2)]})

spec_heating = pd.DataFrame({#'period': np.arange(11),

                            "construction_period": ["before 1919", "1919-1948", "1949-1957", "1958-1968", "1969-1978",
                                                    "1979-1983", "1984-1994", "1995-2001", "2002-2009",
                                                    "2010-2015", "from 2016"],
                             "Demand_Kwh": [round(res19), round(res48), round(res57),
                                            round(res68), round(res78), round(res83), round(res94),
                                            round(res01), round(res09), round(res15), round(res16)]})


cols = ["green"]
plt.rcParams["figure.figsize"] = (8,6)
a_x = sns.barplot(x='construction_period',y = 'Demand_Kwh',
                  width=0.25,data=spec_heating, palette=cols)
for i in a_x.containers:
    a_x.bar_label(i,)
plt.xlabel("Construction Year Period", fontsize=12)
plt.ylabel("Enery demand for space heating [KWh/m^2]", fontsize=12)
plt.yticks([0, 50, 100, 150, 200, 250],fontsize=12)
plt.title("Space heating demand for SFH according to construction year period", fontsize=12)
plt.xticks(rotation=45)
plt.ticklabel_format()
plt.tight_layout()
plt.savefig(f'output/Space Heating Demand for SFH According to Construction Year Period')
plt.show()


# Aggregated heat flow cal for number of buildings
# SFH
res19 = round(building.iloc[0][2] * heating_demand19["sfh"]) # demand 4 sfh b4 1919
res48 = round(building.iloc[1][2] * heating_demand48["sfh"])  # demand 4 sfh b4 1948
res57 = round(building.iloc[2][2] * heating_demand57["sfh"])  # demand 4 sfh b4 1957
res68 = round(building.iloc[3][2] * heating_demand68["sfh"])  # demand 4 sfh b4 1968
res78 = round(building.iloc[4][2] * heating_demand78["sfh"])  # demand 4 sfh b4 1978
res83 = round(building.iloc[5][2] * heating_demand83["sfh"])  # demand 4 sfh b4 1983
res94 = round(building.iloc[6][2] * heating_demand94["sfh"])  # demand 4 sfh b4 1994
res01 = round(building.iloc[7][2] * heating_demand01["sfh"])  # demand 4 sfh b4 2001
res09 = round(building.iloc[8][2] * heating_demand09["sfh"])  # demand 4 sfh b4 2009
res15 = round(building.iloc[9][2] * heating_demand15["sfh"])  # demand 4 sfh b4 2015
res16 = round(building.iloc[10][2] * heating_demand16["sfh"])  # demand 4 sfh b4 2016

res_sfh = pd.DataFrame({"res19":res19, "res48":res48, "res57":res57, "res68":res68, "res78":res78, "res83":res83,
                        "res94":res94, "res01":res01, "res09":res09, "res15":res15, "res16":res16}, index=None)


SFH_sum = []
for i in range(0, len(res_sfh)):
    sfh_sum = ((res_sfh["res19"][i] + res_sfh["res48"][i] + res_sfh["res57"][i] + res_sfh["res68"][i] + res_sfh["res78"][i]
               + res_sfh["res83"][i] + res_sfh["res94"][i] + res_sfh["res01"][i] + res_sfh["res09"][i] + res_sfh["res15"][i]
               + res_sfh["res16"][i]))
    SFH_sum.append(round(sfh_sum))

#res_sfh.to_csv(("output/res_sfh.csv"), index=False)

res_sfh_aggre = pd.DataFrame({"SFH": SFH_sum}, index=None) # result for all period and number of buildings per hours
res_sfh_aggre.to_csv(("output/res_sfh_aggre.csv"), index=False)
#res_all.to_csv(("output/all_res.csv"), index=False)
#print(heating_demand57)


# TH
re19 = round((building.iloc[0][3] * heating_demand19["th"]))  # demand 4 th b4 1919
re48 = round((building.iloc[1][3] * heating_demand48["th"]))  # demand 4 th b4 1948
re57 = round((building.iloc[2][3] * heating_demand57["th"]))  # demand 4 th b4 1957
re68 = round((building.iloc[3][3] * heating_demand68["th"]))  # demand 4 th b4 1968
re78 = round((building.iloc[4][3] * heating_demand78["th"]))  # demand 4 th b4 1978
re83 = round((building.iloc[5][3] * heating_demand83["th"]))  # demand 4 th b4 1983
re94 = round((building.iloc[6][3] * heating_demand94["th"]))  # demand 4 th b4 1994
re01 = round((building.iloc[7][3] * heating_demand01["th"]))  # demand 4 th b4 2001
re09 = round((building.iloc[8][3] * heating_demand09["th"]))  # demand 4 th b4 2009
re15 = round((building.iloc[9][3] * heating_demand15["th"]))  # demand 4 th b4 2015
re16 = round((building.iloc[10][3] * heating_demand16["th"]))  # demand 4 th b4 2016

re_th = pd.DataFrame({"res19":re19, "res48":re48, "res57":re57, "res68":re68, "res78":re78, "res83":re83,
                        "res94":re94, "res01":re01, "res09":re09, "res15":re15, "res16":re16}, index=None)

TH_sum = []
for i in range(0, len(re_th)):
    th_sum = ((re_th["res19"][i] + re_th["res48"][i] + re_th["res57"][i] + re_th["res68"][i] + re_th["res78"][i]
               + re_th["res83"][i] + re_th["res94"][i] + re_th["res01"][i] + re_th["res09"][i] + re_th["res15"][i]
               + re_th["res16"][i]))
    TH_sum.append(round(th_sum))
#print(sum(TH_sum))


# MFH
rmf19 = round((building.iloc[0][4] * heating_demand19["mfh"])) # demand 4 mfh b4 1919
rmf48 = round((building.iloc[1][4] * heating_demand48["mfh"]) )  # demand 4 mfh b4 1948
rmf57 = round((building.iloc[2][4] * heating_demand57["mfh"]) )  # demand 4 mfh b4 1957
rmf68 = round((building.iloc[3][4] * heating_demand68["mfh"]) )  # demand 4 mfh b4 1968
rmf78 = round((building.iloc[4][4] * heating_demand78["mfh"]) )  # demand 4 mfh b4 1978
rmf83 = round((building.iloc[5][4] * heating_demand83["mfh"]))  # demand 4 mfh b4 1983
rmf94 = round((building.iloc[6][4] * heating_demand94["mfh"]))  # demand 4 mfh b4 1994
rmf01 = round((building.iloc[7][4] * heating_demand01["mfh"]))  # demand 4 mfh b4 2001
rmf09 = round((building.iloc[8][4] * heating_demand09["mfh"]))  # demand 4 mfh b4 2009
rmf15 = round((building.iloc[9][4] * heating_demand15["mfh"]))  # demand 4 mfh b4 2015
rmf16 = round((building.iloc[10][4] * heating_demand16["mfh"]))  # demand 4 mfh b4 2016

rmf = pd.DataFrame({"res19":rmf19, "res48":rmf48, "res57":rmf57, "res68":rmf68, "res78":rmf78, "res83":rmf83,
                        "res94":rmf94, "res01":rmf01, "res09":rmf09, "res15":rmf15, "res16":rmf16}, index=None)

MFH_sum = []
for i in range(0, len(re_th)):
    rmf_sum = ((rmf["res19"][i] + rmf["res48"][i] + rmf["res57"][i] + rmf["res68"][i] + rmf["res78"][i]
               + rmf["res83"][i] + rmf["res94"][i] + rmf["res01"][i] + rmf["res09"][i] + rmf["res15"][i]
               + rmf["res16"][i]))
    MFH_sum.append(round(rmf_sum))
#print(sum(TH_sum))

#res_sfh.to_csv(("output/res_sfh.csv"), index=False)

# AB
rab19 = round((building.iloc[0][5] * heating_demand19["ab"])) # demand 4 ab b4 1919
rab48 = round((building.iloc[1][5] * heating_demand48["ab"]))  # demand 4 ab b4 1948
rab57 = round((building.iloc[2][5] * heating_demand57["ab"]))  # demand 4 ab b4 1957
rab68 = round((building.iloc[3][5] * heating_demand68["ab"]))  # demand 4 ab b4 1968
rab78 = round((building.iloc[4][5] * heating_demand78["ab"]))  # demand 4 ab b4 1978
rab83 = round((building.iloc[5][5] * heating_demand83["ab"]))  # demand 4 ab b4 1983
rab94 = round((building.iloc[6][5] * heating_demand94["ab"]))  # demand 4 ab b4 1994
rab01 = round((building.iloc[7][5] * heating_demand01["ab"]))  # demand 4 ab b4 2001
rab09 = round((building.iloc[8][5] * heating_demand09["ab"]))  # demand 4 ab b4 2009
rab15 = round((building.iloc[9][5] * heating_demand15["ab"]))  # demand 4 ab b4 2015
rab16 = round((building.iloc[10][5] * heating_demand16["ab"]))  # demand 4 ab uj,  b4 2016

rab = pd.DataFrame({"res19":rab19, "res48":rab48, "res57":rab57, "res68":rab68, "res78":rab78, "res83":rab83,
                        "res94":rab94, "res01":rab01, "res09":rab09, "res15":rab15, "res16":rab16}, index=None)

AB_sum = []
for i in range(0, len(re_th)):
    ab_sum = ((rab["res19"][i] + rab["res48"][i] + rab["res57"][i] + rab["res68"][i] + rab["res78"][i]
               + rab["res83"][i] + rab["res94"][i] + rab["res01"][i] + rab["res09"][i] + rab["res15"][i]
               + rab["res16"][i]))
    AB_sum.append(round(ab_sum))
#print(sum(AB_sum))

#res_sfh.to_csv(("output/res_sfh.csv"), index=False)

# Aggregated values for all building types

overall_aggre = pd.DataFrame({'period': np.arange(len(SFH_sum)),
                        "SFH": SFH_sum, "TH": TH_sum, "MFH": MFH_sum, "AB": AB_sum}).set_index('period') # hourly

#overall_sum_aggre = pd.DataFrame({"Total Heat Energy": ["SFH", "TH", "MFH", "AB", "Aggregated Demand For Space Heating"],
 #                           "Value[GWh_per_year]": [round(sum(SFH_sum) * 0.000001), round(sum(TH_sum) * 0.000001),
  #                                            round(sum(MFH_sum) * 0.000001),round(sum(AB_sum) * 0.000001),
   #                                           round((sum(SFH_sum) + sum(TH_sum) + sum(MFH_sum) + sum(AB_sum))) * 0.000001]}, index=None)

overall_sum_aggre = pd.DataFrame({"Total Heat Energy": ["SFH", "TH", "MFH", "AB", "Aggregated Demand For Space Heating"],
                            "Value[kWh_per_year]": [round(sum(SFH_sum)), round(sum(TH_sum)),
                                              round(sum(MFH_sum)),round(sum(AB_sum)),
                                              round((sum(SFH_sum) + sum(TH_sum) + sum(MFH_sum) + sum(AB_sum)))]}, index=None)
overall_sum_aggre.to_csv(("output/overall aggregate.csv"), index=False)
overall_aggre.to_csv(("output/overall.csv"), index=False)   # hourly
print(overall_sum_aggre)

#  Aggregated heat demand for entire building types and total number of buildings
flow_sum = pd.DataFrame({"Power_KWh_per_year": [overall_sum_aggre.iloc[0][1], overall_sum_aggre.iloc[1][1],
                                                overall_sum_aggre.iloc[2][1],  overall_sum_aggre.iloc[3][1], overall_sum_aggre.iloc[4][1]],
                         "Building Type":  ["SFH", "TH", "MFH", "AB", "Overall Demand"]}, index=None)
print(flow_sum)
# Reference area for the entire building types
A_ref_sfh = [141.8, 302.5, 111.1, 121.2, 173.2, 215.6, 150.2,121.9,146.5, 186.8, 186.8]
A_ref_th = [96, 112.8, 149.6, 117.4, 100.3, 108.3, 127.6, 148.8, 151.9, 195.8, 195.8]
A_ref_mfh = [312.4, 385.0,  632.3, 3129.1, 468.6, 654.0, 778.1,  834.9, 2190.1, 1305.0, 1305.0]
A_ref_ab = [829.4, 1484.0, 1602.7, 3887.4,  3322.0, 3107.5,  3107.5, 0, 0, 0, 0]
sfh_aggr_area = []
th_aggr_area = []
mfh_aggr_area = []
ab_aggr_area = []

# Reference area multiplied by total number of buildings
for i in range(len(building)):
    sfh_aggr_area.append(A_ref_sfh[i] * building.iloc[i][2])    # sfh
    th_aggr_area.append(A_ref_th[i] * building.iloc[i][3])      # th
    mfh_aggr_area.append(A_ref_mfh[i] * building.iloc[i][4])    # mfh
    ab_aggr_area.append(A_ref_ab[i] * building.iloc[i][5])      # ab

# Heating demand Aggregate per sqm area
flow_sqm = pd.DataFrame({"Power_KWh_per_sqm_year": [round((overall_sum_aggre.iloc[0][1]) / sum(sfh_aggr_area)),
                                                    round((overall_sum_aggre.iloc[1][1]) / sum(th_aggr_area)),
                                                    round((overall_sum_aggre.iloc[2][1]) / sum(mfh_aggr_area)),
                                                    round((overall_sum_aggre.iloc[3][1]) / sum(ab_aggr_area)),
                                                    round((overall_sum_aggre.iloc[4][1])/(sum(sfh_aggr_area)+
                                                                                          sum(th_aggr_area)+
                                                                                          sum(mfh_aggr_area)+sum(ab_aggr_area)))
],

                         "Building Type":  ["SFH", "TH", "MFH", "AB", "Overall Demand"]}, index=None)

#round((overall_sum_aggre.iloc[0][1] / sum(sfh_aggr_area)) +
 #     (overall_sum_aggre.iloc[1][1] / sum(th_aggr_area)) +
  #    (overall_sum_aggre.iloc[2][1] / sum(mfh_aggr_area)) +
   #   (overall_sum_aggre.iloc[3][1] / sum(ab_aggr_area)))
print(flow_sqm["Power_KWh_per_sqm_year"][0] + flow_sqm["Power_KWh_per_sqm_year"][1] + flow_sqm["Power_KWh_per_sqm_year"][2]+
                                                                                flow_sqm["Power_KWh_per_sqm_year"][3])

print(flow_sqm)
cols = ['darkgreen' if (x > 180) else 'darkred' for x in flow_sqm.Power_KWh_per_sqm_year]
plt.rcParams["figure.figsize"] = (8,6)
#fontweight = 'bold'
a_x = sns.barplot(x = 'Building Type',y = 'Power_KWh_per_sqm_year',data = flow_sqm,
                  palette=cols, width=0.25)

for i in a_x.containers:
    a_x.bar_label(i,)

plt.title("Total space heating per square meter area in Thuringia", fontsize = 14)
plt.xlabel("Building Types", fontsize=14)
plt.ylabel("Power_kWh_per_sqm_per_year", fontsize=14)
plt.yticks([0,25, 50, 75, 100, 125, 150, 175, 200],fontsize=12)
plt.ticklabel_format()
plt.tight_layout()
plt.savefig(f'output/Overall Aggregate Demand sqm_T')
plt.show()



flow_pie = [overall_sum_aggre.iloc[0][1], overall_sum_aggre.iloc[1][1], overall_sum_aggre.iloc[2][1],
            overall_sum_aggre.iloc[3][1]]
Building_Type = ["SFH", "TH", "MFH", "AB"]
fig, ax1 = plt.subplots()
ax1.pie(flow_pie, labels=Building_Type, autopct='%1.1f%%',
        colors=['gold', 'green', 'violet', 'blue'],
        textprops={'size': 'smaller'}, radius=0.7)
plt.title("The total percentage share of heat demand in Thuringia",fontsize=14)
plt.ticklabel_format()
plt.tight_layout()
#plt.grid()
plt.savefig(f'output/Overall Demand in percentage')
plt.show()

# Overall Aggregated demand
#print(flow_sum)
cols = ['darkgreen' if (x > 130) else 'darkred' for x in flow_sum.Power_KWh_per_year]
plt.rcParams["figure.figsize"] = (8,6)
a_x2 = sns.barplot(x = 'Building Type',y = 'Power_KWh_per_year',data = flow_sum, palette=cols)
for i in a_x2.containers:
    a_x2.bar_label(i,)
plt.title("Aggregated Heat Demand In Thuringia",fontsize = 14)
plt.ticklabel_format()
plt.tight_layout()
#plt.grid()
plt.savefig(f'output/aggregated Heat demand in thuringia')
plt.show()

fig, ax = plt.subplots()
#fig = plt.rcParams["figure.figsize"] = (10,6)
ax.stackplot(overall_aggre.index, overall_aggre["TH"]*0.001, overall_aggre["MFH"]*0.001,
              overall_aggre["AB"]*0.001, overall_aggre["SFH"]*0.001, colors=['red', 'limegreen', 'purple', 'yellow'],
             labels=['TH', 'MFH', 'AB', 'SFH'])
plt.ylim(0, None)
plt.xlim(0, len(overall_aggre))
plt.xlabel("Time Period (h)",fontsize=12)
plt.title(f' Overall space heating demand profile', fontsize=12)
plt.ylabel('Power (MW)', fontsize = 18)
plt.yticks([0,2000, 4000, 6000, 8000, 10000, 12000, 14000],fontsize=12)
plt.xticks([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016],
      ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'], fontsize=12)
plt.legend(loc="upper right")
#plt.grid()
plt.ticklabel_format()
plt.tight_layout()
plt.savefig(f'output/overall heat profile')
plt.tight_layout()
plt.show()

# TODO Yearly Duration Curve
overall_aggre['Heatlost'] = overall_aggre[list(overall_aggre.columns)].sum(axis=1) # Add the two columns together
heating_demand_array = (overall_aggre['Heatlost']*0.001).values  # sorting values from large to small
hourly_array = np.arange(len(overall_aggre), dtype=int)         # lenght of heating period

Lost_data = {'hours': hourly_array, 'Heatlost': heating_demand_array}
Lost_df = pd.DataFrame(Lost_data)                         # create a datafram of sorted values

sns.set(rc={"figure.figsize":(10, 6)})
p = sns.lineplot(x = "hours", y = "Heatlost", data = Lost_df)
plt.ylim(0, None)
plt.xlim(0, len(overall_aggre))
p.set_title(f"Load duration curve for thuringia", fontsize = 14)
p.set_xlabel("Time (hours)", fontsize = 14)
p.set_ylabel("Power (MW)", fontsize = 14)
plt.xticks([0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016],
      ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])
plt.ticklabel_format()
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig(f'output/year load profile overall')
plt.show()

Lost_df['interval'] = 1
Lost_df_sorted = Lost_df.sort_values(by=['Heatlost'], ascending=False)
Lost_df_sorted['Duration'] = Lost_df_sorted['interval'].cumsum()
Lost_df_sorted['Percentage'] = Lost_df_sorted['Duration']*100/len(overall_aggre)

p = sns.lineplot(x = "Duration", y = "Heatlost", data = Lost_df_sorted)
plt.ylim(0, None)
plt.xlim(0, len(overall_aggre))
p.set_title(f"Load duration curve for residential buildings in Thuringia", fontsize=18)
p.set_xlabel("Time (hours)",fontsize=18)
p.set_ylabel("Power (kW)", fontsize=18)
plt.yticks([0,2000, 4000, 6000, 8000, 10000, 12000, 14000],fontsize=18)
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], fontsize=18)
 #     ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])
plt.tight_layout()
plt.legend()
plt.grid(color='red')
plt.savefig(f'output/overall_duration_curve')
plt.show()
