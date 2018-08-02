from feateng import *
import numpy as np

def test_kfold_mean_encoding():
    """
    test for reg_mean_encoding function
    """
    # test without alpha
    df = pd.DataFrame({"A": [1, 1, 1, 2, 2, 2, 3, 4],
                       "B": [1, 1, 2, 2, 3, 3, 4, 4],
                       "y": [1, 0, 1, 0, 0, 1, 1, 0]})

    # single column test alpha = 0
    out_df = kfold_mean_encoding(df, "A", "encode", "y", splits=2)
    true = [1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.75, 0.75]
    assert true == list(out_df["encode"].values)

    # single column test alpha = 2
    out_df = kfold_mean_encoding(df, "A", "encode", "y", splits=2, alpha=2)
    true = [0.67, 0.67, 0.5, 0.67, 0.67, 0.25, 0.57, 0.57]
    assert np.isclose(np.array(true), np.round(out_df["encode"].values, 2)).all()

    # multi column test alpha = 0
    out_df = kfold_mean_encoding(df, ["A", "B"], "encode", "y", splits=2)
    true = [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5]
    assert true == list(out_df["encode"].values)

    # multi column test alpha = 2
    out_df = kfold_mean_encoding(df, ["A", "B"], "encode", "y", splits=2, alpha=2)
    true = [0.5, 0.5, 0.5, 0.5, 0.67, 0.33, 0.5, 0.5]
    assert np.isclose(np.array(true), np.round(out_df["encode"].values, 2)).all()

    print("Test Passed !")


def test_loo_mean_encoding():
    """
    test for reg_mean_encoding function
    """
    # test without alpha
    df = pd.DataFrame({"A": [1, 1, 1, 2, 2, 2, 3, 4],
                       "B": [1, 1, 2, 2, 3, 3, 4, 4],
                       "y": [1, 0, 1, 0, 0, 1, 1, 0]})

    # single column test alpha = 0
    out_df = loo_mean_encoding(df, "A", "encode", "y", alpha=0)
    true = [0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
    assert true == list(out_df["encode"].values)

    # single column test alpha = 2
    out_df = loo_mean_encoding(df, "A", "encode", "y", alpha=2)
    true = [0.5, 0.75, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5]
    assert true == list(out_df["encode"].values)

    # multi column test alpha = 0
    out_df = loo_mean_encoding(df, ["A", "B"], "encode", "y", alpha=0)
    true = [0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5]
    assert true == list(out_df["encode"].values)

    # multi column test alpha = 2
    out_df = loo_mean_encoding(df, ["A", "B"], "encode", "y", alpha=2)
    true = [0.33, 0.67, 0.5, 0.5, 0.67, 0.33, 0.5, 0.5]
    assert np.isclose(np.array(true), np.round(out_df["encode"].values, 2)).all()

    print("Test Passed !")
    


def test_kfold_count_encoding():
    """
    test for reg_mean_encoding function
    """
    df = pd.DataFrame({"A":[1,1,1,2,2,2,3],
              "B":[1,1,2,2,3,3,4],
              "y":[1,0,1,0,0,1,1]})
    
    # single column test
    out_df = kfold_count_encoding(df, "A", "encode", "y", splits=2, seed=10)
    true = [1,2,1,2,1,1,1]
    assert true == list(out_df["encode"].values)
    
    # multi column test
    out_df = kfold_count_encoding(df, ["A", "B"], "encode", "y", splits=2, seed=10)
    true = [1,1,1,1,1,1,1]
    assert true == list(out_df["encode"].values)
    
    print("Test Passed !")
    
    
