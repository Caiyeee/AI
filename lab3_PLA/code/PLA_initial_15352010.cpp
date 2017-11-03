#include<bits/stdc++.h>
using namespace std;

vector< vector<double> > x;
vector<int> y;
vector<double> w;
vector<int> val_y;
vector<int> res_y;
fstream f1,f2;

int main(){
	//读取训练集
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab3(PLA)\\lab3数据\\thur78train.csv",ios::in);
	string line,s;
	while(getline(f1,line)){
		vector<double> x_line;
		x_line.push_back(1);
		int begin = 0;
		int dot = line.find(",",begin);
		 
		while(dot != -1){
			s = line.substr(begin,dot-begin);
			x_line.push_back(stod(s));
			
			begin = dot+1;
			dot = line.find(",",begin);
		}
		x.push_back(x_line);
		
		s = line.substr(begin,dot-begin);
		y.push_back(stoi(s));
	}
	f1.close();

	for(int i=0; i<x[0].size(); i++)//w初始化
		w.push_back(1);

	//遍历样本，计算w 
	for(int i=0; i<x.size(); i++){
		double sum = 0;
		for(int j=0; j<x[i].size(); j++)//计算w的转置和x的乘积
				sum += x[i][j] * w[j];
				
		while(sum*y[i] <= 0)  { //得到的结果和y不一样 
			for(int j=0; j<w.size(); j++)
				w[j] = w[j] + y[i]*x[i][j];
			
			for(int j=0; j<x[i].size(); j++) //计算w的转置和x的乘积 
				sum += x[i][j] * w[j];
		} 
	}
	cout << "w: ";
	for(int i=0; i<w.size(); i++)
		cout << " " << w[i];
	cout << endl;

	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab3(PLA)\\lab3数据\\test.csv",ios::in);
	f2.open("E:\\学习\\大三上\\人工智能\\实验\\lab3(PLA)\\lab3数据\\15352010_caiye_PLA.csv",ios::out);
	while(getline(f1,line)){
		//读取 
		vector<double> test_line;
		test_line.push_back(1);
		int begin = 0;
		int dot = line.find(",",begin);
		 
		while(dot != -1){
			s = line.substr(begin,dot-begin);
			test_line.push_back(stod(s));
			
			begin = dot+1;
			dot = line.find(",",begin);
		}
		s = line.substr(begin,dot-begin);
//		val_y.push_back(stoi(s));//验证集才需要 
		
		//预测
		double sum = 0;
		for(int i=0; i<w.size(); i++)
			sum += w[i] * test_line[i];
		int res = sum>0 ? +1 : -1;
		res_y.push_back(res);
		f2 << res << endl;
//		cout << res << endl;
	}
	f1.close();
	f2.close();
	
	//计算准确率、召回率、精确率、F值 
//	double TP=0, FN=0, TN=0, FP=0;
//	for(int i=0; i<val_y.size(); i++){
//		if(val_y[i]==1 && res_y[i]==1)
//			TP += 1;
//		else if(val_y[i]==1 && res_y[i]==-1)
//			FN += 1;
//		else if(val_y[i]==-1 && res_y[i]==-1)
//			TN += 1;
//		else if(val_y[i]==-1 && res_y[i]==1)
//			FP += 1;
//	}
//	double Accuracy = (TP+TN) / (TP+FN+TN+FP);
//	double Recall = TP / (TP+FN);
//	double Precision = TP / (TP+FP);
//	double F1 = (2*Precision*Recall) / (Precision+Recall);
//	cout << "TP:" << TP << "  FN:" << FN << "  TN:" << TN << "  FP:"  << FP << endl;
//	cout << "Accuracy: " << Accuracy << endl;
//	cout << "Recall: " << Recall << endl;
//	cout << "Precision: " << Precision << endl;
//	cout << "F1: " << F1 << endl; 
	 
	return 0;
}
