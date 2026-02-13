from tests_new.utils.interfaces.partitioned_table import PartitionedPandasTable
import pandas as pd
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

class TitanicAsDiseaseDataset(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(CURRENT_DIR / "data/titanic_as_disease.parquet")
    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # ⚠ ATTENTION DO NOT DELETE (FUTURE REFERENCE) ⚠
    # ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠
    # def get_dataset(self) -> pd.DataFrame:
    #     import seaborn as sns
    #     from sklearn.preprocessing import LabelEncoder
    #     df = sns.load_dataset('titanic')
    #     df = df[['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'fare']].dropna()
    #     # Treatment: sex (female=1, male=0)
    #     df['Treatment'] = LabelEncoder().fit_transform(df['sex'])  # female=1, male=0
    #     # Outcome: survived
    #     df['Outcome'] = df['survived']
    #     df.to_parquet(CURRENT_DIR / "data/titanic_as_disease.parquet")
    #     return df

if __name__ == "__main__":
    print(TitanicAsDiseaseDataset().get_dataset())