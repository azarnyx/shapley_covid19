# About

The repository represents example on how to analyse SHAP values of pytorch model based on resnet and trained on COVID19 lungs X-Rays dataset (https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).

# Preparation

1. Install requirements
2. Modify "shap" package in the following way:

change line 198 after shap installation in python3.7/site-packages/shap/explainers/_deep/

From:

phis[l][j] = (torch.from_numpy(sample_phis[l][self.data[l].shape[0]:]).to(self.device) * (X[l][j: j + 1] - self.data[l])).cpu().numpy().mean(0)

To:

"phis[l][j] = (torch.from_numpy(sample_phis[l][self.data[l].shape[0]:]).to(self.device) * (X[l][j: j + 1] - self.data[l])).cpu().detach().numpy().mean(0)"

3. Download the Kaggle dataset and move COVID-19 Radiography Database folder to data/ folder
4. Run the script and enjoy shapley explanations
