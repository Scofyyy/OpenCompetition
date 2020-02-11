import unittest
import pandas as pd
from src.tabular.feature_engineering.feature_generator.entity_embedding import entity_embedding

class MyTestCase(unittest.TestCase):
    def test_something(self):
        df_train = pd.read_csv('data/train.csv').loc[:200000, :]

        df = df_train[
            ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
             'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_09_cat', 'ps_car_10_cat',
             'ps_car_11_cat', "target"]]

        conffiger = """
                {
                    "target_col":"target", 
                    "embedding_file_path":"ee_embeddings.pickle" ,
                    "check_point_file_path":"best_model_weights.hdf5",
                    "loss_func":"binary_crossentropy",
                    "optimizer":"adam",
                    "epochs":2,
                    "batch_size": 128,
                    "embed_cols":
                        {
                        "ps_ind_02_cat":{"output_dim":3},
                        "ps_ind_04_cat":{"output_dim":2},
                        "ps_ind_05_cat":{"output_dim":5},
                        "ps_car_01_cat":{"output_dim":7},
                        "ps_car_02_cat":{"output_dim":2},
                        "ps_car_03_cat":{"output_dim":2},
                        "ps_car_04_cat":{"output_dim":5},
                        "ps_car_05_cat":{"output_dim":2},
                        "ps_car_06_cat":{"output_dim":8},
                        "ps_car_07_cat":{"output_dim":2},
                        "ps_car_09_cat":{"output_dim":3},
                        "ps_car_10_cat":{"output_dim":2},
                        "ps_car_11_cat":{"output_dim":21}
                        }

                }
        """
        entity_embedding(df, conffiger)

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
