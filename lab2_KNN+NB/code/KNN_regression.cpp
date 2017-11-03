#include<bits/stdc++.h>
using namespace std;

map<string,int> words;//词汇表及单词的出现顺序 
vector< map<string,double> > lines;//训练集各个样例的单词及其对应的tf矩阵的值 
vector< vector<double> > label;//训练集每个样例的情感标签 
vector< map<string,double> > test;//测试集的单词及其对应的tf矩阵的值 
vector< vector<double> > predict;//预测结果 
fstream f1;

double toDouble(string s){
	int len = s.size();
	double res = 0;
	if(s[0]=='0' && s[1]=='.'){
		for(int i=len-1; i>=2; i--)
			res = (res + s[i] - '0') * 0.1;
	}
	else if(s[0]=='1')
		res = 1;
	return res;
}

int main(){
	//将训练集的数据提取出来 
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\regression_dataset\\train_set.csv",ios::in);
	string line,s;
	getline(f1,line);
	
	while(getline(f1,line)) {
		map<string,double> wordsOfLine;
		int dot = line.find(",");//逗号的位置 
		int begin = 0;
		int end = line.find(" ",begin);//空格的位置 

		while(1) {
			if(end != -1)
				s = line.substr(begin,end-begin);
			else
				s = line.substr(begin,dot-begin);
			
			
			if(!words.count(s))	{
				words[s] = words.size();
				wordsOfLine[s] = 1;
			}
			else
				wordsOfLine[s] += 1;


		
			if(end != -1) {
				begin = end + 1;
				end = line.find(" ",begin);
			}
			else {
				begin = dot + 1;
				dot = line.find(",",begin); 
				break;
			}		
		}	
		
		vector<double> motion;
		while(1) {
			if(dot != -1)
				s = line.substr(begin,dot-begin);
			else
				s = line.substr(begin,line.size()-begin);
			
			motion.push_back(toDouble(s));
		
			if(dot != -1) {
				begin = dot + 1;
				dot = line.find(",",begin);
			}
			else  
				break;
		}
		lines.push_back(wordsOfLine); 
		label.push_back(motion);
	}
	f1.close();
	
	//提取测试集的数据 
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\regression_dataset\\test_set.csv",ios::in);
//	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\regression_dataset\\validation_set.csv",ios::in);
	getline(f1,line);
	while(getline(f1,line)) {
		int begin = 0;
		int end = line.find(" ",begin);
		int dot = line.find(",",begin);
		
		map<string,double> wordsOfLine;
		while(1) {
			if(end != -1)
				s = line.substr(begin,end-begin);
			else
				s = line.substr(begin,dot-begin);
			
			
			if(!words.count(s))	{
				words[s] = words.size();
				wordsOfLine[s] = 1;
			}
			else
				wordsOfLine[s] += 1;


		
			if(end != -1) {
				begin = end + 1;
				end = line.find(" ",begin);
			}
			else {
				begin = dot + 1;
				dot = line.find(",",begin); 
				break;
			}		
		}	
		test.push_back(wordsOfLine);
	}
	f1.close();
	

	//数据处理及预测
//TF矩阵*************************************TF*************************************************************
	//计算训练集的TF矩阵 
	for(int i=0; i<lines.size(); i++){
		double num = 0;
		for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++)
			num += it->second;
		for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++)
			it->second = it->second / num;		
	}
//TF矩阵*************************************TF*************************************************************	
	
	
	
	for(int row=0; row<test.size(); row++){
//TF矩阵*************************************************************************************************** 
		//计算测试集的TF矩阵 
		int sizeInWords = 0; //第row行测试样例的单词数 
		for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++)
			if(words.count(it->first)) //如果出现不在单词表的单词，不参与计数
				sizeInWords += it->second;
		for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++){
			if(words.count(it->first))
				it->second = it->second / sizeInWords; //计算其对应的TF矩阵的值
		}
//TF矩阵*************************************TF*************************************************************

		
		//一个测试样例
		multimap<double,int> distance;//距离,训练集的第几个样例 
		for(int i=0; i<lines.size(); i++){
			double dis = 0;	
			
//TF矩阵计算距离*************************************TF*************************************************************
			for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++){
				if(!test[row].count(it->first))//不在测试样例中
//					dis += pow(it->second,2); //欧式距离 
					dis += it->second; //曼哈顿距离 
				else
//					dis += pow((it->second - test[row][it->first]),2); 
					dis += abs(it->second - test[row][it->first]);
			}
			for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++)
				if(!lines[i].count(it->first))
					if(words.count(it->first))
//						dis += pow(it->second,2);
						dis += it->second;
						
//			dis = sqrt(dis); //欧氏距离 
			distance.insert(pair<double,int>(dis,i));
//TF矩阵计算距离*************************************TF*************************************************************			
			
			
//onehot矩阵计算距离************************************************************************************************ 
			int common = 0;	 
			for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++){
				if(!lines[i].count(it->first)){
					if(words.count(it->first))
						dis++;
				}
				else
					common++; 		
			}
			dis += (lines[i].size() - common);
			distance.insert(pair<double,int>(dis,i));
			dis = 0;	
//onehot矩阵距离************************************************************************************************ 		
		} 
		
		
		vector<double> motion;
		for(int i=0; i<6; i++){ //六种情绪 
			multimap<double,int>::iterator iter = distance.begin();
			double res = 0.0; 
			if(iter->first == 0) { //最近的距离为0，即存在一模一样的样例 
				for(int k=0; k<6; k++){
					motion.push_back(label[iter->second][k]);
				}
				break;
			}
			else{
				for(int j=0; j<4; j++){	//K的取值			 
					res += label[iter->second][i] / (1.0*iter->first);
					iter++;
				}
			}			
			motion.push_back(res);
			res = 0.0;
		}

		//归一化情感系数 
		double sum = 0.0;
		for(int i=0; i<6; i++)
			sum += motion[i];
		for(int i=0; i<6; i++)
			motion[i] = motion[i] / sum;
		predict.push_back(motion);
	}
	
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\regression_dataset\\15352010_caiye_KNN_regression.csv",ios::out); 
	f1 << "id,anger,disgust,fear,joy,sad,surprise\n";
	for(int i=0; i<predict.size();i++) {
		f1 << i+1;
		for(int j=0; j<6; j++){
				f1 << "," << fixed << setprecision(6) << predict[i][j];	
		}
		f1 << endl;
	}
	f1.close();
	
	return 0;
} 
