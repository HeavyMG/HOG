#pragma once
#include <opencv2/core/mat.hpp>

#define GRADIENT 180
#define EPSILON 1e-6f

class hog
{
public:
	void process(const cv::Mat& image);
	const std::vector<float> retrieve(const cv::Rect& window);
	const cv::Mat get_vector_mask(const int thiccness = 1);
	const cv::Mat& get_mags() const;
	const cv::Mat& get_oris() const;

	void save(const std::string& filename);
	static hog load(const std::string& filename);
	std::vector<float> l2_norm(const std::vector<float>& v);
	std::vector<float> l2_hys(const std::vector<float>& v);

public:
	hog();
	hog(const size_t blocksize, const size_t cellsize, const size_t stride, const size_t binning = 9); // default is 9
	hog(const hog& rhs);
	hog(hog&& rhs) noexcept;
	hog& operator = (const hog& rhs);
	hog& operator = (hog&& rhs) noexcept;
	~hog();

private:
	void extract_mag_and_ori(const cv::Mat& image);
	const std::vector<float> process_cell(const cv::Mat& cell_mag, const cv::Mat& cell_ori) const;

private:
	size_t _blocksize;
	size_t _cellsize;
	size_t _stride;
	size_t _binning;
	size_t _bin_width;

	size_t _n_cells_per_block_y = _blocksize / _cellsize;
	size_t _n_cells_per_block_x = _n_cells_per_block_y;
	size_t _n_cells_per_block = _n_cells_per_block_y * _n_cells_per_block_x;
	size_t _block_hist_size = _binning * _n_cells_per_block;
	size_t _stride_unit = _stride / _cellsize;
	size_t _n_cells_y;
	size_t _n_cells_x;

	cv::Mat mag;
	cv::Mat ori;
	std::vector<std::vector<std::vector<float>>> _cell_hists;
};
