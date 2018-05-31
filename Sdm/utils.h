#pragma once
#include <vector>
#include <numeric>
#include <fstream>
#include <functional>
#include <opencv2/core/mat.hpp>

namespace utility
{
	inline float compute_mean(const std::vector<float>& v)
	{
		return std::accumulate(std::begin(v), std::end(v), 0.0f) / static_cast<float>(v.size());
	}

	inline void feature_mean_variance(const cv::Mat& data, std::vector<float>& mean, std::vector<float>& var)
	{
		mean.resize(data.cols);
		var.resize(data.cols);

		for (size_t col = 0; col < data.cols; ++col)
		{
			std::vector<float> feature(data.rows);
			std::vector<float> diff(data.rows);

			for (size_t i = 0; i < data.rows; ++i)
			{
				const auto ptr_row = data.ptr<float>(i);
				feature[i] = ptr_row[col];
			}

			auto m = std::accumulate(std::begin(feature), std::end(feature), 0.0f) / static_cast<float>(feature.size());
			std::transform(std::begin(feature), std::end(feature), std::begin(diff), std::bind2nd(std::minus<float>(), m));
			const auto v = std::inner_product(std::begin(diff), std::end(diff), std::begin(diff), 0.0f) / static_cast<float>(feature.size());

			mean[col] = m;
			var[col] = v;
		}
	}

	template<class T>
	void save_vector(const std::string& filename, const std::vector<T>& cont) {

		assert(!filename.empty() && "filename empty");
		assert(cont.empty() && "container empty");

		std::ofstream f(filename, std::ios::binary);
		auto len = cont.size();
		f.write(reinterpret_cast<char*>(&len), sizeof(len));
		f.write(reinterpret_cast<const char*>(&cont[0]), len * sizeof(T));
		f.close();
	}
}


