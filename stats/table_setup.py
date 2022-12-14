import pandas as pd

columns = [
"METHOD",
"acc",
"f1",
"auc",
"stat_parity_race",
"stat_parity_gender",
"stat_parity_age",
"equal_opp_race",
"equal_opp_gender",
"equal_opp_age",
"equal_opp_race_gender",
"acc_race0",
"acc_race1",
"acc_race2",
"acc_gender0",
"acc_gender1",
"acc_gender2",
"acc_age0",
"acc_age1",
"acc_age2",
"acc_age3",
"acc_age4",
"acc_race0_gender0",
"acc_race0_gender1",
"acc_race0_gender2",
"acc_race1_gender0",
"acc_race1_gender1",
"acc_race1_gender2",
"acc_race2_gender0",
"acc_race2_gender1",
"acc_race2_gender2",
"acc_race0_age0",
"acc_race0_age1",
"acc_race0_age2",
"acc_race0_age3",
"acc_race0_age4",
"acc_race1_age0",
"acc_race1_age1",
"acc_race1_age2",
"acc_race1_age3",
"acc_race1_age4",
"acc_race2_age0",
"acc_race2_age1",
"acc_race2_age2",
"acc_race2_age3",
"acc_race2_age4",
"acc_gender0_age0",
"acc_gender0_age1",
"acc_gender0_age2",
"acc_gender0_age3",
"acc_gender0_age4",
"acc_gender1_age0",
"acc_gender1_age1",
"acc_gender1_age2",
"acc_gender1_age3",
"acc_gender1_age4",
"acc_gender2_age0",
"acc_gender2_age1",
"acc_gender2_age2",
"acc_gender2_age3",
"acc_gender2_age4",
"acc_race0_gender0_age0",
"acc_race0_gender0_age1",
"acc_race0_gender0_age2",
"acc_race0_gender0_age3",
"acc_race0_gender0_age4",
"acc_race0_gender1_age0",
"acc_race0_gender1_age1",
"acc_race0_gender1_age2",
"acc_race0_gender1_age3",
"acc_race0_gender1_age4",
"acc_race0_gender2_age0",
"acc_race0_gender2_age1",
"acc_race0_gender2_age2",
"acc_race0_gender2_age3",
"acc_race0_gender2_age4",
"acc_race1_gender0_age0",
"acc_race1_gender0_age1",
"acc_race1_gender0_age2",
"acc_race1_gender0_age3",
"acc_race1_gender0_age4",
"acc_race1_gender1_age0",
"acc_race1_gender1_age1",
"acc_race1_gender1_age2",
"acc_race1_gender1_age3",
"acc_race1_gender1_age4",
"acc_race1_gender2_age0",
"acc_race1_gender2_age1",
"acc_race1_gender2_age2",
"acc_race1_gender2_age3",
"acc_race1_gender2_age4",
"acc_race2_gender0_age0",
"acc_race2_gender0_age1",
"acc_race2_gender0_age2",
"acc_race2_gender0_age3",
"acc_race2_gender0_age4",
"acc_race2_gender1_age0",
"acc_race2_gender1_age1",
"acc_race2_gender1_age2",
"acc_race2_gender1_age3",
"acc_race2_gender1_age4",
"acc_race2_gender2_age0",
"acc_race2_gender2_age1",
"acc_race2_gender2_age2",
"acc_race2_gender2_age3",
"acc_race2_gender2_age4",
]
df = pd.DataFrame(columns=columns)
df.to_csv("./stats/table.csv", index=False)