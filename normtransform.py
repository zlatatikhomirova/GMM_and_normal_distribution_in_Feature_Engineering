class NormTransform:
    def __init__(self):
        sqrt_inv = lambda x: x ** 2
        log_inv = lambda x: np.exp(x)
        inv_inv = lambda x: 1 / x
        ! !boxcox_inv = ...

    @staticmethod
    def is_normally_distribured(x: np.ndarray|pd.Series, pct_threshold: float=0.7) -> tuple((bool, pd.DataFrame)):
        norm_distr_check_df = norm_distr_check(x, p_level=p_level)
        not_accepted = norm_distr_check_df.conclusion[norm_distr_check_df.conclusion.str.contains(r'not')]
        return len(not_accepted) / len(x) >= pct_threshold, norm_distr_check_df
    
    @staticmethod
    def is_multimodal(x: np.ndarray|pd.Series, p_value_threshold: float=0.001):
        # Тест Хартигана-Диппеля
        dip, p_value = diptest(x)
        return p_value <= p_value_threshold

    @staticmethod
    def cluster_distr_gmm(x: np.ndarray|pd.Series, n: int) -> sklearn.mixture.GaussianMixture:
        train = x.values if x.dtype == pd.Series else x
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(train.reshape(-1, 1))
        return gmm

    @staticmethod
    def find_optimal_gmm(x: np.ndarray|pd.Series, maxn_cluster: int) -> np.ndarray:
        pass

    @staticmethod    
    def inv_transform(x: np.ndarray|pd.Series) -> tuple((np.ndarray, )):
        return 1 / x, 

    @staticmethod
    def log_transform(x: np.ndarray|pd.Series) -> np.ndarray:
        return np.log(x)

    @staticmethod
    def sqrt_transform(x: np.ndarray|pd.Series) -> np.array:
        return np.sqrt(x)

    @staticmethod
    def boxcox_transform(x: np.ndarray|pd.Series, retpar: bool=False):
        lmax_pearsonr_Y, lmax_mle_Y = sps.boxcox_normmax(x, method='all')
        Y_boxcox = sps.boxcox(x, lmbda=(lmax_mle_Y))
        if retpar:
            return lmax_mle_Y, Y_boxcox
        return Y_boxcox
