import cv2
import numpy as np


class HOGExtractor:
    def __init__(self, window_size=(64, 64), cell_size=8, block_size=2, orientations=9):
        self.window_size = window_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.num_bins = orientations
        self.orientations = orientations
        cells_per_dim = window_size[0] // cell_size
        blocks_per_dim = cells_per_dim - block_size + 1
        self.feature_dim = blocks_per_dim * blocks_per_dim * block_size * block_size * orientations
        self.bin_width = 180.0 / orientations

    def extract(self, image):
        gray = self._preprocess(image)
        gx, gy = self._compute_gradients(gray)
        magnitude, orientation = self._compute_magnitude_orientation(gx, gy)
        cell_histograms = self._build_cell_histograms(magnitude, orientation)
        features = self._block_normalize(cell_histograms)
        return features

    def extract_with_visualization(self, image):
        gray = self._preprocess(image)
        gx, gy = self._compute_gradients(gray)
        magnitude, orientation = self._compute_magnitude_orientation(gx, gy)
        cell_histograms = self._build_cell_histograms(magnitude, orientation)
        features = self._block_normalize(cell_histograms)
        hog_image = self._visualize_hog(cell_histograms, gray.shape)
        return features, hog_image

    def _preprocess(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = cv2.resize(gray, self.window_size)
        gray = gray.astype(np.float64) / 255.0
        return gray

    def _compute_gradients(self, gray):
        h, w = gray.shape
        gx = np.zeros((h, w), dtype=np.float64)
        gy = np.zeros((h, w), dtype=np.float64)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        return gx, gy

    def _compute_magnitude_orientation(self, gx, gy):
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.rad2deg(np.arctan2(gy, gx)) % 180
        return magnitude, orientation

    def _build_cell_histograms(self, magnitude, orientation):
        h, w = magnitude.shape
        cells_y = h // self.cell_size
        cells_x = w // self.cell_size
        cs = self.cell_size
        mag_cropped = magnitude[:cells_y * cs, :cells_x * cs]
        ori_cropped = orientation[:cells_y * cs, :cells_x * cs]
        mag_reshaped = mag_cropped.reshape(cells_y, cs, cells_x, cs)
        ori_reshaped = ori_cropped.reshape(cells_y, cs, cells_x, cs)
        mag_cells = mag_reshaped.transpose(0, 2, 1, 3).reshape(cells_y, cells_x, cs * cs)
        ori_cells = ori_reshaped.transpose(0, 2, 1, 3).reshape(cells_y, cells_x, cs * cs)
        bin_idx = (ori_cells / self.bin_width) - 0.5
        bin_idx = bin_idx % self.num_bins
        lower_bin = np.floor(bin_idx).astype(np.int32) % self.num_bins
        upper_bin = (lower_bin + 1) % self.num_bins
        upper_weight = bin_idx - np.floor(bin_idx)
        lower_weight = 1.0 - upper_weight
        histograms = np.zeros((cells_y, cells_x, self.num_bins))
        for cy in range(cells_y):
            for cx in range(cells_x):
                mag = mag_cells[cy, cx]
                lb = lower_bin[cy, cx]
                ub = upper_bin[cy, cx]
                lw = lower_weight[cy, cx]
                uw = upper_weight[cy, cx]
                hist = np.bincount(lb, weights=mag * lw, minlength=self.num_bins)
                hist += np.bincount(ub, weights=mag * uw, minlength=self.num_bins)
                histograms[cy, cx] = hist[:self.num_bins]
        return histograms

    def _block_normalize(self, cell_histograms):
        cells_y, cells_x, _ = cell_histograms.shape
        blocks_y = cells_y - self.block_size + 1
        blocks_x = cells_x - self.block_size + 1
        features = []
        eps = 1e-5
        for by in range(blocks_y):
            for bx in range(blocks_x):
                block = []
                for dy in range(self.block_size):
                    for dx in range(self.block_size):
                        block.extend(cell_histograms[by + dy, bx + dx, :])
                block_vector = np.array(block, dtype=np.float64)
                norm = np.sqrt(np.sum(block_vector**2) + eps**2)
                block_vector = block_vector / norm
                block_vector = np.minimum(block_vector, 0.2)
                norm = np.sqrt(np.sum(block_vector**2) + eps**2)
                block_vector = block_vector / norm
                features.extend(block_vector)
        return np.array(features, dtype=np.float64)

    def _visualize_hog(self, cell_histograms, image_shape):
        cells_y, cells_x, num_bins = cell_histograms.shape
        vis_image = np.zeros(image_shape, dtype=np.float64)
        bin_angles = np.arange(num_bins) * self.bin_width + self.bin_width / 2
        bin_angles_rad = bin_angles * np.pi / 180
        for cy in range(cells_y):
            for cx in range(cells_x):
                hist = cell_histograms[cy, cx, :]
                center_y = cy * self.cell_size + self.cell_size // 2
                center_x = cx * self.cell_size + self.cell_size // 2
                for bin_idx in range(num_bins):
                    mag = hist[bin_idx]
                    angle = bin_angles_rad[bin_idx]
                    max_mag = np.max(cell_histograms) + 1e-6
                    length = (mag / max_mag) * (self.cell_size // 2)
                    dx = length * np.cos(angle)
                    dy = length * np.sin(angle)
                    x1, y1 = int(center_x - dx), int(center_y - dy)
                    x2, y2 = int(center_x + dx), int(center_y + dy)
                    cv2.line(vis_image, (x1, y1), (x2, y2), 1.0, 1)
        return vis_image

    def get_info(self):
        return {'window_size': self.window_size, 'cell_size': self.cell_size, 'block_size': self.block_size, 'orientations': self.orientations, 'feature_dimension': self.feature_dim}
