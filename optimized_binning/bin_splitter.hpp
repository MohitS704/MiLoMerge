#ifndef binSplitter
#define binSplitter

#include <unordered_map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#if EIGEN_VERSION_AT_LEAST(3,4,0)
    #define Run
#endif

#ifndef Run
#error Bin_splitter needs Eigen >= 3.4.0
#endif

typedef Eigen::Array<long double, Eigen::Dynamic, Eigen::Dynamic> ArrayXXld;
typedef Eigen::Array<long double, Eigen::Dynamic, 1> ArrayXld;

class bin_splitter{
    private:
        std::vector<Eigen::ArrayXXd> data;
        size_t nObservables;
        size_t nPoints;
        size_t nHypotheses;
        std::vector<std::string> encodedFinalStrings;
        std::vector<int> finalBinCounts;
        std::vector<int> hypoList;
        std::vector<int> observablesList;
        std::vector<std::pair<double,double>> maximaAndMinima;
        std::unordered_map<std::string, std::vector<std::vector<int>>> bins;
        
        void initialize(
            std::vector<std::vector<std::vector<double>>>& data,
            std::vector<std::vector<double>>& weights
        );
        
        void score(
            std::vector<std::vector<int>>& b1, 
            std::vector<std::vector<int>>& b2,
            long double& metricVal,
            bool compareToFirstOnly
        );

    public:
        bin_splitter(
            std::vector<std::vector<std::vector<double>>>& data
        );

        bin_splitter(
            std::vector<std::vector<std::vector<double>>>& data,
            std::vector<double>& weight
        );

        bin_splitter(
            std::vector<std::vector<std::vector<double>>>& data,
            std::vector<std::vector<double>>& weights
        );

        Eigen::MatrixXd getData(size_t h);


        void split(
            size_t nBinsDesired,
            size_t granularity,
            double statLimit
        );

        void reset();

        ~bin_splitter() noexcept;
};

#endif
