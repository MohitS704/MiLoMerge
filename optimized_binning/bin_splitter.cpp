#include <bin_splitter.hpp>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>
#include <fstream>
#include <utility>
#include <memory>
#include <string>
#include <limits>

bin_splitter::bin_splitter(
    std::vector<std::vector<std::vector<double>>>& data,
    std::vector<std::vector<double>>& weights
){
    this->nPoints = data[0][0].size();
    this->nObservables = data[0].size();
    this->nHypotheses = data.size();

    initialize(
        data,
        weights
    );
}

bin_splitter::bin_splitter(
    std::vector<std::vector<std::vector<double>>>& data
){
    this->nPoints = data[0][0].size();
    this->nObservables = data[0].size();
    this->nHypotheses = data.size();

    std::vector<std::vector<double>> weights;
    for(int i = 0; i < this->nHypotheses; i++){
        weights.push_back(std::vector<double>(this->nPoints));
        std::fill(weights[i].begin(), weights[i].end(), 1);
    }

    initialize(
        data,
        weights
    );
}

bin_splitter::bin_splitter(
    std::vector<std::vector<std::vector<double>>>& data,
    std::vector<double>& weight
){
    this->nPoints = data[0][0].size();
    this->nObservables = data[0].size();
    this->nHypotheses = data.size();

    std::vector<std::vector<double>> weights;
    for(int i = 0; i < this->nHypotheses; i++){
        weights.push_back(std::vector<double>(this->nPoints));
        std::fill(weights[i].begin(), weights[i].end(), weight[i]);
    }

    initialize(
        data,
        weights
    );
}

void bin_splitter::initialize(
    std::vector<std::vector<std::vector<double>>>& data,
    std::vector<std::vector<double>>& weights
){
    this->hypoList = std::vector<int>(this->nHypotheses);
    this->observablesList = std::vector<int>(this->nObservables);
    this->encodedFinalStrings = std::vector<std::string>();
    this->finalBinCounts = std::vector<int>();
    
    //matrices of size (this->nPoints, this->nObservables + 1)
    this->data = std::vector<Eigen::ArrayXXd>(this->nHypotheses);
    
    //the other part of the matrix holds all the weights!
    this->bins = std::unordered_map<
        std::string, 
        std::vector<std::vector<int>>
    >();
    //Eigen matrices are column-major!!
    this->maximaAndMinima = std::vector<std::pair<double,double>>(
        this->nObservables
    );
    

    int h = 0;
    int i = 0;
    int j = 0;
    for(h = 0; h < this->nHypotheses; h++){
        (this->data)[h] = Eigen::ArrayXXd(
            this->nPoints,
            this->nObservables + 1
        );
    }

    for(j = 0; j < this->nObservables; j++){
        this->maximaAndMinima[j] = std::make_pair(
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()
        );
    }

    std::vector<std::vector<int>> starting_bin;

    for (h = 0; h < this->nHypotheses; h++){
        starting_bin.push_back(std::vector<int>());
        for (j = 0; j < this->nObservables; j++){
            for(i = 0; i < this->nPoints; i++){
                ((this->data)[h])(i,j) = data[h][j][i];
                // std::cout << h << " " << j << " " << i << std::endl;
            }
            
        }
        for(i = 0; i < this->nPoints; i++){
            (starting_bin)[h].push_back(i);
            ((this->data)[h])(i,this->nObservables) = weights[h][i];
        }
        (this->hypoList)[h] = h;

        Eigen::VectorXd minima = ((this->data)[h]).colwise().minCoeff();
        Eigen::VectorXd maxima = ((this->data)[h]).colwise().maxCoeff();
        for(j = 0; j < this->nObservables; j++){
            if(minima(j) < this->maximaAndMinima[j].first){
                this->maximaAndMinima[j].first = minima(j);
            }
            if(maxima(j) > this->maximaAndMinima[j].second){
                this->maximaAndMinima[j].second = maxima(j);
            }
        }
    }

    this->bins.emplace("", starting_bin);
}

Eigen::MatrixXd bin_splitter::getData(size_t h){
    return (this->data)[h];
}

void bin_splitter::score(
    std::vector<std::vector<int>>& b1,
    std::vector<std::vector<int>>& b2,
    long double& metricVal,
    bool compareToFirstOnly
){
    metricVal = 0;
    std::vector<int> hypothesisTopLoop;
    if(compareToFirstOnly){
        hypothesisTopLoop.push_back(0);
    } else{
        hypothesisTopLoop = this->hypoList;
    }

    long double t1,t2,t3,t4 = 0;
    for(int h : hypothesisTopLoop){
        Eigen::VectorXd hVec = ((this->data)[h]).col(this->nObservables);
        //maps to the weight for that hypo
        for(int hPrime = h + 1; hPrime < this->nHypotheses; hPrime++){
            Eigen::VectorXd hPrimeVec = ((this->data)[hPrime]).col(this->nObservables);
            t1 = hVec(b1[h]).sum();
            t2 = hPrimeVec(b2[hPrime]).sum();
            t3 = hVec(b2[h]).sum();
            t4 = hPrimeVec(b1[hPrime]).sum();

            metricVal += (
                t1*t2*t1*t2 +
                t3*t4*t3*t4 -
                2*t1*t2*t3*t4
            );
        }
    }
}

void bin_splitter::split(
    size_t nBinsDesired,
    size_t granularity,
    double statLimit
){
    Eigen::ArrayXXd possibleEdges = Eigen::ArrayXXd(granularity, this->nObservables);
    ArrayXXld scores = ArrayXXld(granularity, this->nObservables);
    Eigen::Index maxRow, maxCol;

    for(int i = 0; i < this->nObservables; i++){
        possibleEdges.col(i) = Eigen::ArrayXd::LinSpaced(
            granularity + 2,
            this->maximaAndMinima[i].first,
            this->maximaAndMinima[i].second
        )(Eigen::seq(1, granularity)); //utilizing the edges is useless!
    }
    size_t nBins = 1;
    while (nBins < nBinsDesired){
        const size_t nLeaves = (this->bins).size();
        ArrayXld scoresPerParent = ArrayXld(nLeaves);

        std::vector<std::pair<int,int>> obsAndEdgeIndexPerParent(nLeaves);
        std::vector<std::string> encodedCutsPerParent(nLeaves);

        size_t parentCounter = 0;
        for(auto it : this->bins){ //all the possible leaf nodes
            scores.setZero(); //reset scores for this parent
            for(int obs = 0; obs < this->nObservables; obs++){
                for(int edgeIndex = 0; edgeIndex < granularity; edgeIndex++){
                    std::vector<std::vector<int>> b1;
                    std::vector<std::vector<int>> b2;
                    double cut = possibleEdges.coeff(edgeIndex, obs);
                    long double score = 0;

                    std::cout << "Trying cut of " << obs << " > " << cut << std::endl;
                    std::cout << "At index: (" << edgeIndex << "," << obs << ")" << std::endl;
                    std::cout << "On possible parent " << parentCounter << ": " << it.first << std::endl;
                    size_t h = 0;
                    bool breakLoop = false;
                    for(std::vector<int> hypothesisIndices : it.second){
                        if(breakLoop){
                            break;
                        }
                        for(int index : hypothesisIndices){
                            b1.push_back(std::vector<int>());
                            b2.push_back(std::vector<int>());
                            if(((this->data)[h]).coeff(index, obs) > cut){
                                b1[h].push_back(index);
                            } else{
                                b2[h].push_back(index);
                            }
                        }
                        if(b1[h].size() < statLimit | b2[h].size() < statLimit){
                            breakLoop = true;
                            continue;
                        }
                        h++;
                    }
                    if(breakLoop){
                        continue;
                    }
                    this->score(b1, b2, scores(edgeIndex, obs), false);
                    std::cout << "score of: " << scores.coeff(edgeIndex, obs) << std::endl;
                }
            }
            scoresPerParent(parentCounter) = scores.maxCoeff(&maxRow, &maxCol);
            std::cout << "picked best score at index: " << maxRow << "," << maxCol << std::endl;
            obsAndEdgeIndexPerParent[parentCounter] = std::make_pair(maxRow, maxCol);
            encodedCutsPerParent[parentCounter] = it.first;
            parentCounter++;
        }
        if(!scoresPerParent.all()){
            std::cerr << "No more cuts can be applied." << std::endl;
            std::cerr << "Ending prematurely at " << nBins << " bins." << std::endl;
            break;
        }

        std::cout << "scores per parent: " << scoresPerParent << std::endl;
        long double overallMaximum = scoresPerParent.maxCoeff(&maxCol);
        std::cout << "best index is: " << maxCol << std::endl;
        int chosenObs = obsAndEdgeIndexPerParent[maxCol].second;
        int chosenEdgeIndex = obsAndEdgeIndexPerParent[maxCol].first;
        std::string encodedCut = encodedCutsPerParent[maxCol];

        std::vector<std::vector<int>> b1;
        std::vector<std::vector<int>> b2;
        double cut = possibleEdges.coeff(chosenEdgeIndex, chosenObs);
        std::cout << "Picked cut: " << chosenObs << " > " << cut << std::endl;
        std::cout << "with a score of " << overallMaximum << std::endl;

        int h = 0;
        for(std::vector<int> hypothesisIndices : (this->bins)[encodedCut]){
            b1.push_back(std::vector<int>());
            b2.push_back(std::vector<int>());
            for(int index : hypothesisIndices){
                if(((this->data)[h]).coeff(index, chosenObs) > cut){
                    b1[h].push_back(index);
                } else{
                    b2[h].push_back(index);
                }
            }
            h++;
        }

        //old cut is no longer a leaf
        //replace it with this!

        (this->bins).erase(encodedCut);
        (this->bins).emplace(
            encodedCut + ";" + std::to_string(chosenObs) + ">" + std::to_string(cut),
            b1
        );
        (this->bins).emplace(
            encodedCut + ";" + std::to_string(chosenObs) + "<=" + std::to_string(cut),
            b2
        );
        std::cout << std::endl << "Done with bin " << nBins << std::endl << std::endl;
        nBins++;
    }
    
    std::ofstream cutsFile;
    cutsFile.open("cuts.log");
    for(auto it: this->bins){
        this->encodedFinalStrings.push_back(it.first);
        this->finalBinCounts.push_back(it.second.size());
        cutsFile << it.first << "\n";
    }
    cutsFile.close();
}

void bin_splitter::reset(){
    this->bins.clear();
    this->encodedFinalStrings.clear();
    std::vector<std::vector<int>> starting_bin;

    for (int h = 0; h < this->nHypotheses; h++){
        starting_bin.push_back(std::vector<int>());
        for(int i = 0; i < this->nPoints; i++){
            (starting_bin)[h].push_back(i);
        }
    }
    this->bins.emplace("", starting_bin);
}

bin_splitter::~bin_splitter() noexcept{
    this->bins.clear();
    this->encodedFinalStrings.clear();
    this->data.clear();
    this->finalBinCounts.clear();
    this->hypoList.clear();
    this->observablesList.clear();
    this->maximaAndMinima.clear();
    this->bins.clear();
}