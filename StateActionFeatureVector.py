import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.tile_width = tile_width
        self.num_tilings = num_tilings
        self.num_tiles = np.ceil((state_high - state_low) / tile_width) + 1
        self.num_actions = num_actions

        self.tilings_start = np.empty((num_tilings, len(tile_width)), float)
        self.total_tiles = int(np.prod(self.num_tiles))

        for i in range(0, num_tilings):
            start = (state_low - i / self.num_tilings * tile_width)
            self.tilings_start[i] = start

        self.tile_weights = np.zeros((self.num_tilings * self.num_actions * self.total_tiles), float)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """

        return self.num_actions * self.num_tilings * self.total_tiles

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        features = np.zeros((self.num_actions * self.num_tilings * self.total_tiles), float)
        if done:
            return features
        for i in range(0, self.num_tilings):
            tile = (s - self.tilings_start[i]) // self.tile_width
            features[int(i * self.total_tiles + tile[0] * self.num_tiles[1] + tile[
                1] + a)] = 1.0  # storing tiles as a 1 d array, doing index manipultion to access the correct tile
        return features
