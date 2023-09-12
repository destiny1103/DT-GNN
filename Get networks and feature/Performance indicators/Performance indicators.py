import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

plt.rcParams['font.family'] = 'Times New Roman'

# Input (training data and predictions) Output (confusion matrix)
AIS_path = '.\\Trajectory_label&features.csv'
pred_path = '.\\ChebNet-bestpred1.csv'

AIS_df = pd.read_csv(AIS_path)
AIS_df['Label'] = AIS_df['Label'].apply(lambda x: 2 if x == 5 else x)
pred_df = pd.read_csv(pred_path, header=None)
pred_df.columns = ['pred_df']
datum = pd.concat([AIS_df, pred_df], axis=1)

Berth_Berth = datum.loc[(datum['Label'] == 2) & (datum['pred_df'] == 2)]['count'].sum()
Berth_Anchorage = datum.loc[(datum['Label'] == 2) & (datum['pred_df'] == 1)]['count'].sum()
Berth_Moving = datum.loc[(datum['Label'] == 2) & (datum['pred_df'] == 0)]['count'].sum()
Anchorage_Berth = datum.loc[(datum['Label'] == 1) & (datum['pred_df'] == 2)]['count'].sum()
Anchorage_Anchorage = datum.loc[(datum['Label'] == 1) & (datum['pred_df'] == 1)]['count'].sum()
Anchorage_Moving = datum.loc[(datum['Label'] == 1) & (datum['pred_df'] == 0)]['count'].sum()
Moving_Berth = datum.loc[(datum['Label'] == 0) & (datum['pred_df'] == 2)]['count'].sum()
Moving_Anchorage = datum.loc[(datum['Label'] == 0) & (datum['pred_df'] == 1)]['count'].sum()
Moving_Moving = datum.loc[(datum['Label'] == 0) & (datum['pred_df'] == 0)]['count'].sum()
# print('Berth:', Berth_Berth + Berth_Anchorage + Berth_Moving)
# print('Anchorage:', Anchorage_Berth + Anchorage_Anchorage + Anchorage_Moving)
# print('Moving:', Moving_Berth + Moving_Anchorage + Moving_Moving)

# create confusion matrix
y_true = np.array([-1] * (Berth_Berth + Berth_Anchorage + Berth_Moving) +
                  [0] * (Anchorage_Berth + Anchorage_Anchorage + Anchorage_Moving) +
                  [1] * (Moving_Berth + Moving_Anchorage + Moving_Moving))
y_pred = np.array([-1] * Berth_Berth + [0] * Berth_Anchorage + [1] * Berth_Moving +
                  [-1] * Anchorage_Berth + [0] * Anchorage_Anchorage + [1] * Anchorage_Moving +
                  [-1] * Moving_Berth + [0] * Moving_Anchorage + [1] * Moving_Moving)
# print(y_true)
# print(y_pred)
# print(len(y_pred))

cm = confusion_matrix(y_true, y_pred)
# print(cm)
totals = cm.sum(axis=1)
conf_matrix_pct = pd.DataFrame(np.round(cm / totals[:, None] * 100, decimals=2),
                               index=['Berth', 'Anchorage', 'Moving'],
                               columns=['Berth', 'Anchorage', 'Moving'])


print('Accuracy:', accuracy_score(y_true, y_pred))
print('------Weighted------')
print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
print('------Macro------')
print('Macro precision', precision_score(y_true, y_pred, average='macro'))
print('Macro recall', recall_score(y_true, y_pred, average='macro'))
print('Macro f1-score', f1_score(y_true, y_pred, average='macro'))
print('------Micro------')
print('Micro precision', precision_score(y_true, y_pred, average='micro'))
print('Micro recall', recall_score(y_true, y_pred, average='micro'))
print('Micro f1-score', f1_score(y_true, y_pred, average='micro'))

# plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
h = sns.heatmap(conf_matrix_pct, annot=True, annot_kws={"size": 31}, cmap=plt.get_cmap('Blues'), fmt='.2f', linewidths=6)

# Get the colorbar object and set the font size.
cbar = h.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
# confusion_out_path = r'---\\ChebNet-confusion.png'
# plt.savefig(confusion_out_path, bbox_inches='tight', dpi=600)


