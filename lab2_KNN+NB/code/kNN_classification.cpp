#include<bits/stdc++.h>
using namespace std;

map<string,int> words;//词汇表及单词的出现顺序 
vector< map<string,double> > lines;//训练集各个样例的单词及其TF矩阵对应的值 
vector<string> label;//训练集每个样例的情感标签 
vector< map<string,double> > test;//测试集的单词及其对应的TF矩阵的值 
vector<string> validation;//测试集的标签 
vector<string> predict;//预测结果 
fstream f1;
 

int main()
{
	//将训练集的数据提取出来 
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\classification_dataset\\train_set.csv",ios::in);
	string line,s;
	getline(f1,line);
	
	while(getline(f1,line)) {
		map<string,double> wordsOfLine;
		int dot = line.find(",");//逗号的位置 
		int begin = 0;
		int end = line.find(" ",begin);//空格的位置 

		while(1) {//读取每个单词 
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
				s = line.substr(dot+1,line.size()-dot-1);//读取情感标签 
				label.push_back(s); 
				break;
			}
				
		}	
		lines.push_back(wordsOfLine); 
	}
	f1.close();
	
	
	
	//提取测试集的数据 
//	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\zhousixiawuKNN\\test_set.csv",ios::in);
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\classification_dataset\\test_set.csv",ios::in);
	getline(f1,line);
	while(getline(f1,line)) {
		map<string,double> m;
		int begin = 0;
		int end = line.find(" ",begin);
		int dot = line.find(",",begin);
		
		while(1) {
			if(end != -1)
				s = line.substr(begin,end-begin);
			else
				s = line.substr(begin,dot-begin);
				
			
			if(!m.count(s))
				m[s] = 1;
			else 
				m[s] += 1;
			
			
			if(end != -1){
				begin = end + 1;
				end = line.find(" ",begin);
			}
			else {
				//验证集时才需要下面两行 
				s = line.substr(dot+1,line.size()-dot-1);
				validation.push_back(s);
				break;
			}
		}
		test.push_back(m);
	}
	f1.close();
	
	
	//数据处理及预测
//TF矩阵begin*************************************TF*************************************************************
	//计算训练集的TF矩阵 
	for(int i=0; i<lines.size(); i++){
		double num = 0;//单词总个数 
		for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++)
			num += it->second;
		for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++)
			it->second = it->second / num;		
	}
//TF矩阵end*************************************TF*************************************************************



	//遍历每个测试样例
	for(int row=0; row<test.size(); row++){		
//TF矩阵begin*************************************TF*************************************************************
		int sizeInWords = 0; //第row行测试样例的单词数 
		for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++)
			if(words.count(it->first)) //如果出现不在单词表的单词，不参与计数 
				sizeInWords += it->second;
		for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++){
			if(words.count(it->first))
				it->second = it->second / sizeInWords; //计算其对应的TF矩阵的值
		}
//TF矩阵end*************************************TF*************************************************************
		
		multimap<double,int> distance;//TF矩阵的距离,训练集的第几个样例 		
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
//			int common = 0;	 
//			for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++){
//				if(!lines[i].count(it->first)){
//					if(words.count(it->first))
//						dis++;
//				}
//				else
//					common++; 		
//			}
//			dis += (lines[i].size() - common);
//			distance.insert(pair<double,int>(dis,i));
//			dis = 0;
//onehot矩阵计算距离************************************************************************************************ 			
		}


		map<string,int> motion;
		motion["joy"] = 0;
		motion["fear"] = 0;
		motion["sad"] = 0;
		motion["anger"] = 0;
		motion["surprise"] = 0;
		motion["disgust"] = 0;
		
		//找到前k个数的众数，得到其label 
		multimap<double,int>::iterator iter=distance.begin();
		for(int num=0; num<3; num++){ //k的取值 
			motion[label[(iter++)->second]] += 1;
		}
		int max=-1;
		string pred;
		bool flag = 0;
		for(map<string,int>::iterator it=motion.begin(); it!=motion.end(); it++){
			if(it->second > max){
				max = it->second;
				pred = it->first;
				flag = 0;
			}
			else if(it->second == max)
				flag = 1;
		}
		if(flag) {//不存在唯一的最大值时，选用距离最近的label 
			//取属于众数的情况中距离最近的那个label
			for(multimap<double,int>::iterator it=distance.begin(); it!=distance.end(); it++){
				if(motion[pred] == motion[label[it->second]]){
					predict.push_back(label[it->second]);
					break;
				}
			}
//			predict.push_back(label[distance.begin()->second]);//取最近距离的label 
		}			
		else
			predict.push_back(pred); 	
		
	}
	
	
//测试集结果**********************************************************************	
//	把预测结果写进文件
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\classification_dataset\\15352010_caiye_KNN_classification.csv",ios::out); 
	f1 << "textid,label\n";
	for(int i=0; i<predict.size();i++) {
		f1 << i+1 << "," << predict[i] << endl;
	}
	f1.close(); 
//测试集结果********************************************************************** 


//验证集结果********************************************************************** 
//	对比，计算准确率 
//	int num = 0;
//	for(int i=0; i<predict.size(); i++){
////		cout << validation[i] << " " <<  predict[i] << endl;
//		if(predict[i] == validation[i])
//			num++;
//	} 
//	cout << num*1.0/(predict.size()*1.0) << endl;
//验证集结果************************************************************************ 

	return 0;
}
