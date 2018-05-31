#pragma once
#include "hog.h"

#include "opencv2/ml/ml.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <numeric>

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
	assert(!cont.empty() && "container empty");

	std::ofstream f(filename, std::ios::binary);
	auto len = cont.size();
	f.write(reinterpret_cast<char*>(&len), sizeof(len));
	f.write(reinterpret_cast<const char*>(&cont[0]), len * sizeof(T));
	f.close();
}

inline void train_svm(const cv::Mat& mat_data, const cv::Mat& mat_labels)
{
	cv::Ptr<cv::ml::TrainData> dataset = cv::ml::TrainData::create(mat_data, cv::ml::SampleTypes::ROW_SAMPLE, mat_labels);

	cv::Ptr<cv::ml::SVM> clf = cv::ml::SVM::create();
	clf->setType(cv::ml::SVM::C_SVC);
	clf->setKernel(cv::ml::SVM::LINEAR);
	clf->setDegree(2);
	clf->setNu(0.5);
	clf->setC(10);
	clf->setGamma(100);
	clf->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 3000, 1e-12));

	for (int k = 0; k < 3; k++) {

		// random split
		dataset->setTrainTestSplitRatio(0.7, true);

		// training & test samples
		const cv::Mat train_idx = dataset->getTrainSampleIdx();
		const cv::Mat train_data = cv::ml::TrainData::getSubVector(mat_data, train_idx);
		const cv::Mat train_labels = cv::ml::TrainData::getSubVector(mat_labels, train_idx);

		const cv::Mat test_idx = dataset->getTestSampleIdx();
		cv::Mat test_data = cv::ml::TrainData::getSubVector(mat_data, test_idx);
		cv::Mat test_labels = cv::ml::TrainData::getSubVector(mat_labels, test_idx);

		clf->train(train_data, cv::ml::SampleTypes::ROW_SAMPLE, train_labels);

		for (size_t i = 0; i < test_data.rows; ++i)
		{
			const cv::Mat row(1, test_data.cols, CV_32FC1, test_data.ptr<float>(i));
			clf->predict(row);
		}

	}
	std::cout << "Validation done!\n";

	clf->train(dataset->getSamples(), cv::ml::SampleTypes::ROW_SAMPLE, dataset->getResponses());

	clf->save("clf.ext");

	std::cout << "Training done!\n";
}

inline void train2()
{
	// Size of the box that should contain a person
	const cv::Size person_size(40, 120);

	// Set up the HOG for training
	const size_t blocksize = 10;
	const size_t cellsize = 5;
	const size_t stride = 5;
	const size_t binning = 9;

	hog hog(blocksize, cellsize, stride, binning);

	hog.save("hog.ext"); // Save the HOG for faster testing

	std::vector<std::vector<float>> data;
	std::vector<int> labels;

	std::cout << "Starting training for persons" << std::endl;
	for (int i = 0; i < 560; ++i)
	{
		const std::string filename = "./date/persoane/" + std::to_string(i) + ".jpg";
		const auto image = cv::imread(filename, CV_8U);

		if (image.data)
		{
			// Retrieve the HOG from the image
			hog.process(image);
			const auto& hist = hog.retrieve(cv::Rect(0, 0, person_size.width, person_size.height));

			data.push_back(hist);
			labels.push_back(1);
		}
		else
		{
			std::cerr << "invalid image " << filename << std::endl;
		}
	}
	std::cout << "Training for persons done" << std::endl;

	std::cout << "Starting training for non persons" << std::endl;
	for (int i = 0; i < 9120; ++i)
	{
		const std::string filename = "./date/obiecte/" + std::to_string(i) + ".jpg";
		const auto image = cv::imread(filename, CV_8U);

		if (image.data)
		{
			// Retrieve the HOG from the image
			hog.process(image);
			const auto& hist = hog.retrieve(cv::Rect(0, 0, person_size.width, person_size.height));

			data.push_back(hist);
			labels.push_back(-1);
		}
		else
		{
			std::cerr << "invalid image " << filename << std::endl;
		}
	}
	std::cout << "Training for non persons done" << std::endl;

	const cv::Mat mat_labels(labels, false); // convert the labels vector into a matrix
	cv::Mat mat_data(data.size(), data[0].size(), CV_32FC1); // convert the data into matrix

	for (size_t i = 0; i < mat_data.rows; ++i)
	{
		for (size_t j = 0; j < mat_data.cols; ++j)
		{
			const auto val = data[i][j];
			mat_data.at<float>(i, j) = val;
		}
	}

	std::vector<float> mean, var;
	feature_mean_variance(mat_data, mean, var);
	save_vector("mean.ext", mean);
	save_vector("var.ext", var);

	std::cout << "Get mean and variance of the features done!\n";

	// Normalization zero-mean and unit-variance
	for (size_t i = 0; i < mat_data.rows; ++i)
	{
		for (size_t j = 0; j < mat_data.cols; ++j)
		{
			mat_data.at<float>(i, j) -= mean[j];
			mat_data.at<float>(i, j) /= var[j];
		}
	}

	train_svm(mat_data, mat_labels);
}
