#include<bits/stdc++.h>
using namespace std;

map<string,int> words;//�ʻ�����ʵĳ���˳�� 
vector< map<string,double> > lines;//ѵ�������������ĵ��ʼ����Ӧ��tf�����ֵ 
vector< vector<double> > label;//ѵ����ÿ����������б�ǩ 
vector< map<string,double> > test;//���Լ��ĵ��ʼ����Ӧ��tf�����ֵ 
vector< vector<double> > predict;//Ԥ���� 
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
	//��ѵ������������ȡ���� 
	f1.open("E:\\ѧϰ\\������\\�˹�����\\ʵ��\\lab2(KNN+NB)\\DATA\\regression_dataset\\train_set.csv",ios::in);
	string line,s;
	getline(f1,line);
	
	while(getline(f1,line)) {
		map<string,double> wordsOfLine;
		int dot = line.find(",");//���ŵ�λ�� 
		int begin = 0;
		int end = line.find(" ",begin);//�ո��λ�� 

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
	
	//��ȡ���Լ������� 
	f1.open("E:\\ѧϰ\\������\\�˹�����\\ʵ��\\lab2(KNN+NB)\\DATA\\regression_dataset\\test_set.csv",ios::in);
//	f1.open("E:\\ѧϰ\\������\\�˹�����\\ʵ��\\lab2(KNN+NB)\\DATA\\regression_dataset\\validation_set.csv",ios::in);
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
	

	//���ݴ���Ԥ��
//TF����*************************************TF*************************************************************
	//����ѵ������TF���� 
	for(int i=0; i<lines.size(); i++){
		double num = 0;
		for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++)
			num += it->second;
		for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++)
			it->second = it->second / num;		
	}
//TF����*************************************TF*************************************************************	
	
	
	
	for(int row=0; row<test.size(); row++){
//TF����*************************************************************************************************** 
		//������Լ���TF���� 
		int sizeInWords = 0; //��row�в��������ĵ����� 
		for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++)
			if(words.count(it->first)) //������ֲ��ڵ��ʱ�ĵ��ʣ����������
				sizeInWords += it->second;
		for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++){
			if(words.count(it->first))
				it->second = it->second / sizeInWords; //�������Ӧ��TF�����ֵ
		}
//TF����*************************************TF*************************************************************

		
		//һ����������
		multimap<double,int> distance;//����,ѵ�����ĵڼ������� 
		for(int i=0; i<lines.size(); i++){
			double dis = 0;	
			
//TF����������*************************************TF*************************************************************
			for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++){
				if(!test[row].count(it->first))//���ڲ���������
//					dis += pow(it->second,2); //ŷʽ���� 
					dis += it->second; //�����پ��� 
				else
//					dis += pow((it->second - test[row][it->first]),2); 
					dis += abs(it->second - test[row][it->first]);
			}
			for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++)
				if(!lines[i].count(it->first))
					if(words.count(it->first))
//						dis += pow(it->second,2);
						dis += it->second;
						
//			dis = sqrt(dis); //ŷ�Ͼ��� 
			distance.insert(pair<double,int>(dis,i));
//TF����������*************************************TF*************************************************************			
			
			
//onehot����������************************************************************************************************ 
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
//onehot�������************************************************************************************************ 		
		} 
		
		
		vector<double> motion;
		for(int i=0; i<6; i++){ //�������� 
			multimap<double,int>::iterator iter = distance.begin();
			double res = 0.0; 
			if(iter->first == 0) { //����ľ���Ϊ0��������һģһ�������� 
				for(int k=0; k<6; k++){
					motion.push_back(label[iter->second][k]);
				}
				break;
			}
			else{
				for(int j=0; j<4; j++){	//K��ȡֵ			 
					res += label[iter->second][i] / (1.0*iter->first);
					iter++;
				}
			}			
			motion.push_back(res);
			res = 0.0;
		}

		//��һ�����ϵ�� 
		double sum = 0.0;
		for(int i=0; i<6; i++)
			sum += motion[i];
		for(int i=0; i<6; i++)
			motion[i] = motion[i] / sum;
		predict.push_back(motion);
	}
	
	f1.open("E:\\ѧϰ\\������\\�˹�����\\ʵ��\\lab2(KNN+NB)\\DATA\\regression_dataset\\15352010_caiye_KNN_regression.csv",ios::out); 
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
