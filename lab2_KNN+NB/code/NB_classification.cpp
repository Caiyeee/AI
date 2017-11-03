#include<bits/stdc++.h>
using namespace std;


//0：joy，1：sad，2：anger，3：fear，4：disgust，5：surprise，6：total 
map<int,string> motion;
map<string,int> words;//单词表 
vector< map<string,int> > label[6];//不同标签的每一行的词及其出现次数
int total_words_label[7] = {0}; //不同标签中总共出现多少次单词（不去重）
int total_words_norepeat = 0;//训练集的单词表总数 
int total_doc_label[7] = {0};//文章总数，不同标签的文章数 
vector<string> result;//存放预测结果 
vector<string> validation;//验证集的标签 
fstream f1,f2;
	 

int main(){
	motion[0]="joy";
	motion[1]="sad";
	motion[2]="anger";
	motion[3]="fear";
	motion[4]="disgust";
	motion[5]="surprise";
	
	
	//提取训练集数据 
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\lassification_dataset\\train_set.csv",ios::in);
	string line,s;
	getline(f1,line);
	
	while(getline(f1,line)) {
		map<string,int> wordsOfLine;
		int dot = line.find(",");//逗号的位置 
		int begin = 0;
		int end = line.find(" ",begin);//空格的位置 
		
		while(1) {
		if(end != -1)
			s = line.substr(begin,end-begin);
		else
			s = line.substr(begin,dot-begin);
			
		if(!words.count(s))
			words[s] = 1;
		if(!wordsOfLine.count(s))
			wordsOfLine[s] = 1;
		else
			wordsOfLine[s] += 1;


		if(end != -1) {
			begin = end + 1;
			end = line.find(" ",begin);
		}
		else {
			s = line.substr(dot+1,line.size()-dot-1);
			break;
		}
	}	
		if(s == "joy")
			label[0].push_back(wordsOfLine); 
		else if(s == "sad")
			label[1].push_back(wordsOfLine); 
		else if(s == "anger")
			label[2].push_back(wordsOfLine); 
		else if(s == "fear")
			label[3].push_back(wordsOfLine); 
		else if(s == "disgust")
			label[4].push_back(wordsOfLine); 
		else if(s == "surprise")
			label[5].push_back(wordsOfLine); 
	} 
	f1.close();
	
	
	for(int i=0; i<6; i++){
		total_doc_label[i] = label[i].size();
		total_doc_label[6] +=  total_doc_label[i];
	}
	for(int i=0; i<6; i++){
		for(int j=0; j<label[i].size(); j++){
			for(map<string,int>::iterator it=label[i][j].begin(); it!=label[i][j].end(); it++)
				total_words_label[i] += it->second;
		}
	}
	
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\lassification_dataset\\test_set.csv",ios::in);
	f2.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\classification_dataset\\15352010_caiye_NB_classification.csv",ios::out);
	f2 << "textid,label\n";  
	int textid = 0;
	getline(f1,line);
	while(getline(f1,line)) {
		map<string,int> test;
		
		int dot = line.find(",");//逗号的位置 
		int begin = 0;
		int end = line.find(" ",begin);//空格的位置 
		
		while(1) {
			if(end != -1)
				s = line.substr(begin,end-begin);
			else
				s = line.substr(begin,dot-begin);
			
			if(!test.count(s))
				test[s] = 1;
			else
				test[s] += 1;


			if(end != -1) {
				begin = end + 1;
				end = line.find(" ",begin);
			}
			else {
				s = line.substr(dot+1,line.size()-dot-1);
				break;
			}				
		}
		validation.push_back(s);
		
		double predict[6];//六种情绪的预测值 
		for(int i=0; i<6; i++)
			predict[i] = 1;			
		for(int i=0; i<6; i++){//六种情绪 
			vector<int> timeOfWord;//nWi(xk) 单词在情感标签为i的训练样本中出现的次数 
			for(map<string,int>::iterator it=test.begin(); it!=test.end(); it++){
				int cnt = 0;
				for(int j=0; j<label[i].size(); j++){
					if(label[i][j].count(it->first)){
						cnt += label[i][j][it->first];	
					} 	
				}
				timeOfWord.push_back(cnt);
			} 
			
			for(int j=0; j<timeOfWord.size(); j++) //各个p(xk/ei) = (1+nWi(xk)) / (nWi+V)相乘 
				predict[i] *= (1.0*timeOfWord[j]+1)/(1.0*total_words_label[i]+1.0*words.size());
			// P(ei) = Ni/N	
			predict[i] *= (1.0*total_doc_label[i])/(1.0*total_doc_label[6]);
			
		}
		
		double sum = 0; 
		for(int i=0; i<6; i++){			
			sum += predict[i];
		}
			
		for(int i=0; i<6; i++){
			predict[i] = predict[i]/sum;
//			f2 << motion[i] << ":" << predict[i] << endl;
		}
			
		
		
		int max = 0;
		for(int i=1; i<6; i++){
			if(predict[i] > predict[max])
				max = i;
		}
		
		result.push_back(motion[max]);
		
		f2 << ++textid << "," << motion[max] << endl;
	}
	f1.close();
	f2.close();

	//计算准确率	
//	int num = 0;
//	for(int i=0; i<result.size(); i++){
////		cout << validation[i] << " " <<  predict[i] << endl;
//		if(result[i] == validation[i])
//			num++;
//	} 
//	cout << num*1.0/(result.size()*1.0) << endl;
	
	return 0;
} 
