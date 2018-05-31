#include "hog.h"
#include <numeric>
#include <opencv2/shape/hist_cost.hpp>
#include <fstream>

std::vector<float> hog::l2_norm(const std::vector<float>& v)
{
	auto f = v;

	// compute the 2-norm 
	std::transform(v.begin(), v.end(), f.begin(), [](auto val) {return powf(val, 2.f); });

	auto numitor = std::accumulate(f.begin(), f.end(), powf(EPSILON, 2.f));
	numitor = std::sqrt(numitor);

	assert(numitor != 0);

	std::transform(v.begin(), v.end(), f.begin(), [numitor](auto val) {return val / numitor; });

	return f;
}

std::vector<float> hog::l2_hys(const std::vector<float>& v)
{
	auto f = l2_norm(v);
	const auto clip = 0.2f;

	std::transform(f.begin(), f.end(), f.begin(), [](const float & x) {
		if (x > 0.2) return 0.2f;
		else if (x < 0) return 0.0f;
		else return x; });

	return l2_norm(f);
}

void hog::process(const cv::Mat& image)
{
	assert(image.data != nullptr && "image data must not be null");
	for (auto& h1 : _cell_hists) {
		for (auto& h2 : h1)
			h2.clear();
		h1.clear();
	}
	_cell_hists.clear();

	extract_mag_and_ori(image);

	_n_cells_x = static_cast<int>(mag.cols / _cellsize);
	_n_cells_y = static_cast<int>(mag.rows / _cellsize);

	_cell_hists.resize(_n_cells_y);

	for (size_t i = 0; i < _n_cells_y; ++i)
	{
		_cell_hists[i].resize(_n_cells_x);

		for (size_t j = 0; j < _n_cells_x; ++j)
		{
			const auto& cell_rect = cv::Rect(j * _cellsize, i * _cellsize, _cellsize, _cellsize);
			const auto& cell_hist = process_cell(cv::Mat(mag, cell_rect), cv::Mat(ori, cell_rect));
			_cell_hists[i][j] = cell_hist;
		}
	}
}

const std::vector<float> hog::retrieve(const cv::Rect& window)
{
	// add safeguards?

	const size_t x = static_cast<int>(window.x / _cellsize);
	const size_t y = static_cast<int>(window.y / _cellsize);
	const size_t width = static_cast<int>(window.width / _cellsize);
	const size_t height = static_cast<int>(window.height / _cellsize);

	std::vector<float> hog_hist;

	for (size_t block_y = y; block_y <= y + height - _n_cells_per_block_y; block_y += _stride_unit)
	{
		for (size_t block_x = x; block_x <= x + width - _n_cells_per_block_x; block_x += _stride_unit)
		{
			std::vector<float> block_hist;
			block_hist.reserve(_block_hist_size);

			for (size_t cell_y = block_y; cell_y < block_y + _n_cells_per_block_y; ++cell_y)
			{
				for (size_t cell_x = block_x; cell_x < block_x + _n_cells_per_block_x; ++cell_x)
				{
					const std::vector<float>& cell_hist = _cell_hists[cell_y][cell_x];
					block_hist.insert(std::end(block_hist), std::begin(cell_hist), std::end(cell_hist));
				}
			}
			auto f = l2_hys(block_hist);
			hog_hist.insert(hog_hist.end(), f.begin(), f.end());
		}
	}

	return hog_hist;
}

void hog::extract_mag_and_ori(const cv::Mat& image)
{
	cv::Mat dx, dy;
	cv::Sobel(image, dx, CV_32F, 1, 0, 1);
	cv::Sobel(image, dy, CV_32F, 0, 1, 1);
	cv::cartToPolar(dx, dy, mag, ori, true);
}

const std::vector<float> hog::process_cell(const cv::Mat& cell_mag, const cv::Mat& cell_ori) const
{
	std::vector<float> cell_hist(_binning);

	for (size_t i = 0; i < cell_mag.rows; ++i)
	{
		const auto ptr_row_mag = cell_mag.ptr<float>(i);
		const auto ptr_row_ori = cell_ori.ptr<float>(i);

		for (size_t j = 0; j < cell_mag.cols; ++j)
		{
			auto orientation = ptr_row_ori[j];
			if (orientation >= 180)
			{
				orientation -= 180;
			}

			cell_hist.at(static_cast<int>(orientation / _bin_width)) += ptr_row_mag[j];
		}
	}

	return cell_hist;
}

const cv::Mat& hog::get_mags() const
{
	return mag;
}

const cv::Mat& hog::get_oris() const
{
	return ori;
}

const cv::Mat hog::get_vector_mask(const int thiccness)
{
	std::vector<std::vector<float>> cell_hist_maxs(_n_cells_y);
	cv::Mat vector_mask = cv::Mat::zeros(mag.size(), CV_8U);
	float max = 0;

	for (size_t i = 0; i < _n_cells_y; ++i)
	{
		cell_hist_maxs[i].resize(_n_cells_x);

		for (size_t j = 0; j < _n_cells_x; ++j)
		{
			const auto& cell_hist = _cell_hists[i][j];
			const auto& cell_hist_max = *std::max_element(cell_hist.begin(), cell_hist.end());
			cell_hist_maxs[i][j] = cell_hist_max;

			if (cell_hist_max > max)
			{
				max = cell_hist_max;
			}
		}
	}

	for (size_t i = 0; i < _n_cells_y; ++i)
	{
		for (size_t j = 0; j < _n_cells_x; ++j)
		{
			const auto& cell_hist = _cell_hists[i][j];
			const auto color_mag = static_cast<int>(cell_hist_maxs[i][j] / max * 255.0f);

			for (size_t k = 0; k < cell_hist.size(); ++k)
			{
				const auto len = static_cast<int>((cell_hist[k] / cell_hist_maxs[i][j]) * _cellsize / 2);

				if (len > 0)
				{

					cv::line(vector_mask,
						cv::Point(j*_cellsize + _cellsize / 2 + cos((k * _bin_width + 180) * 3.1415 / 180)*len,
							i*_cellsize + _cellsize / 2 + sin((k * _bin_width + 180) * 3.1415 / 180)*len),
						cv::Point(j*_cellsize + _cellsize / 2 + cos((k * _bin_width) * 3.1415 / 180)*len,
							i*_cellsize + _cellsize / 2 + sin((k * _bin_width) * 3.1415 / 180)*len),
						cv::Scalar(color_mag, color_mag, color_mag), thiccness);
				}
			}
			cv::line(vector_mask, cv::Point(j*_cellsize - 1, i*_cellsize - 1), cv::Point(j*_cellsize + mag.rows - 1, i*_cellsize - 1), cv::Scalar(255, 255, 255), thiccness);
			cv::line(vector_mask, cv::Point(j*_cellsize - 1, i*_cellsize - 1), cv::Point(j*_cellsize - 1, i*_cellsize + mag.rows - 1), cv::Scalar(255, 255, 255), thiccness);
		}
	}

	return vector_mask;
}

void hog::save(const std::string& filename)
{
	try {
		std::ofstream f(filename, std::ios::binary);
		f.write(reinterpret_cast<char*>(&_blocksize), sizeof(_blocksize));
		f.write(reinterpret_cast<char*>(&_cellsize), sizeof(_cellsize));
		f.write(reinterpret_cast<char*>(&_stride), sizeof(_stride));
		f.write(reinterpret_cast<char*>(&_binning), sizeof(_binning));
		f.write(reinterpret_cast<char*>(&_bin_width), sizeof(_bin_width));
		f.close();
	}
	catch (...) {
		throw;
	}
}

hog hog::load(const std::string& filename)
{
	size_t blocksize;
	size_t cellsize;
	size_t stride;
	size_t binning;

	try {
		std::ifstream f(filename, std::ios::binary);
		f.read(reinterpret_cast<char*>(&blocksize), sizeof(blocksize));
		f.read(reinterpret_cast<char*>(&cellsize), sizeof(cellsize));
		f.read(reinterpret_cast<char*>(&stride), sizeof(stride));
		f.read(reinterpret_cast<char*>(&binning), sizeof(binning));
		f.close();

		return hog(blocksize, cellsize, stride, binning);
	}
	catch (...) {
		throw;
	}
}

hog::hog()
	:_blocksize(16), _cellsize(16), _stride(8), _binning(9), _bin_width(GRADIENT / _binning)
{}

hog::hog(const size_t blocksize, const size_t cellsize, const size_t stride, const size_t binning)
	: _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(binning), _bin_width(GRADIENT / _binning)
{
	assert(_blocksize > 2);
	assert(_cellsize > 1);
	assert(_binning > 2);
}

hog::~hog() = default;

hog::hog(const hog& rhs)
	:_blocksize(rhs._blocksize),
	_cellsize(rhs._cellsize),
	_stride(rhs._stride),
	_binning(rhs._binning),
	_bin_width(rhs._bin_width)
{
}

hog::hog(hog&& rhs) noexcept
	:_blocksize(rhs._blocksize),
	_cellsize(rhs._cellsize),
	_stride(rhs._stride),
	_binning(rhs._binning),
	_bin_width(rhs._bin_width)
{
}

hog& hog::operator=(const hog& rhs)
{
	if (this != &rhs)
	{
		_blocksize = rhs._blocksize;
		_cellsize = rhs._cellsize;
		_stride = rhs._stride;
		_binning = rhs._binning;
		_bin_width = rhs._bin_width;
	}

	return *this;
}

hog& hog::operator=(hog&& rhs) noexcept
{
	if (this != &rhs)
	{
		_blocksize = rhs._blocksize;
		_cellsize = rhs._cellsize;
		_stride = rhs._stride;
		_binning = rhs._binning;
		_bin_width = rhs._bin_width;
	}

	return *this;
}
