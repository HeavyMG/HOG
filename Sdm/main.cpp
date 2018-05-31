#include "hog.h"
#include "train.h"
#include <vector>
#include <iostream>
#include <conio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/shape/hist_cost.hpp>
#include "opencv2/objdetect.hpp"

template<class T>
auto load_vector(const std::string& filename) {
	try {
		std::vector<T> v;
		std::ifstream f(filename, std::ios::binary);
		unsigned int len = 0;
		f.read(reinterpret_cast<char*>(&len), sizeof(len));
		v.resize(len);
		if (len > 0)
			f.read(reinterpret_cast<char*>(&v[0]), len * sizeof(T));
		f.close();
		return v;
	}
	catch (...) {
		throw;
	}
}


void merge_mats(const cv::Mat& A, const cv::Mat& B, const std::string& name) {
	cv::Mat superimposed;
	cv::addWeighted(A, 0.5, B, 0.5, 0.0, superimposed);
	cv::imshow(name, superimposed);
}

cv::Mat custom_normalization(cv::Mat src) {
	src.convertTo(src, CV_8U);
	return src;
}

int main()
{

	//train2();

	/*const cv::Mat image = cv::imread("person.jpg", CV_8U);
	cv::imshow("original", image);

	size_t blocksize = 64;
	size_t cellsize = 32;
	size_t stride = 64;
	size_t binning = 9;

	hog hog(blocksize, cellsize, stride, binning);
	hog.process(image);
	
	auto hist = hog.retrieve(cv::Rect(0, 0, image.cols, image.rows));

	std::cout << "Histogram size: " << hist.size() << "\n";

	for (auto h : hist)
		std::cout << h << ",";
	std::cout << "\n";
	merge_mats(image, hog.get_vector_mask(2), "vector_mask");
	merge_mats(custom_normalization(hog.get_mags()), hog.get_vector_mask(1), "magnitude");
	merge_mats(custom_normalization(hog.get_oris()), hog.get_vector_mask(1), "orientation");

	cv::waitKey(0);
	*/
	cv::Size window(40, 120);

	// load models
	cv::Ptr<cv::ml::SVM> clf = cv::Algorithm::load<cv::ml::SVM>("clf.ext");
	auto mean = load_vector<float>("mean.ext");
	auto var = load_vector<float>("var.ext");
	hog hog = hog::load("hog.ext");
	cv::Mat image, image2;

	// read video sequence image by image

	std::stringstream filename;
	//filename << "./dataset/iccv07-data/images/pedxing-seq1/" << std::setfill('0') << std::setw(8) << std::to_string(i) << ".jpg";
	image = cv::imread("test2.jpg", CV_32F);
	cv::cvtColor(image, image2, cv::COLOR_BGR2GRAY);
	cv::imshow("original", image);
	hog.process(image2);

	// list of rectangles (positive matches)
	std::vector<cv::Rect> list_rect;

	for (size_t x = 0; x < image.cols - window.width; x += 5) {
		for (size_t y = 0; y < image.rows - window.height; y += 5) {

			cv::Rect rec = cv::Rect(x, y, window.width, window.height);
			auto hist = hog.retrieve(rec);

			// normalization zero-mean unit-variance
			for (size_t k = 0; k < hist.size(); ++k) {
				hist[k] -= mean[k];
				hist[k] /= var[k];
			}

			cv::Mat sample = cv::Mat(1, hist.size(), CV_32FC1, hist.data());
			if (clf->predict(sample) == 1) {
				list_rect.push_back(rec);
			}

		}
	}
	std::cout << "A";
	// non-max suppression
	cv::groupRectangles(list_rect, 3, 9 / 100.f);

	// draw rectangles
	for (auto& rec : list_rect) {
		cv::rectangle(image, rec, cv::Scalar(255, 255, 255), 2);
	}
	cv::imshow("HOG", image);
	cv::waitKey(0);
	//display_orig.next_frame(image);
	//record_video.next_frame(image);



	while (!_kbhit());
	return 0;
}