#include<bits/stdc++.h>
using namespace std;

map<string,int> words;//�ʻ�����ʵĳ���˳�� 
vector< map<string,double> > lines;//ѵ�������������ĵ��ʼ���TF�����Ӧ��ֵ 
vector<string> label;//ѵ����ÿ����������б�ǩ 
vector< map<string,double> > test;//���Լ��ĵ��ʼ����Ӧ��TF�����ֵ 
vector<string> validation;//���Լ��ı�ǩ 
vector<string> predict;//Ԥ���� 
fstream f1;
 

int main()
{
	//��ѵ������������ȡ���� 
	f1.open("E:\\ѧϰ\\������\\�˹�����\\ʵ��\\lab2(KNN+NB)\\DATA\\classification_dataset\\train_set.csv",ios::in);
	string line,s;
	getline(f1,line);
	
	while(getline(f1,line)) {
		map<string,double> wordsOfLine;
		int dot = line.find(",");//���ŵ�λ�� 
		int begin = 0;
		int end = line.find(" ",begin);//�ո��λ�� 

		while(1) {//��ȡÿ������ 
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
				s = line.substr(dot+1,line.size()-dot-1);//��ȡ��б�ǩ 
				label.push_back(s); 
				break;
			}
				
		}	
		lines.push_back(wordsOfLine); 
	}
	f1.close();
	
	
	
	//��ȡ���Լ������� 
//	f1.open("E:\\ѧϰ\\������\\�˹�����\\ʵ��\\lab2(KNN+NB)\\DATA\\zhousixiawuKNN\\test_set.csv",ios::in);
	f1.open("E:\\ѧϰ\\������\\�˹�����\\ʵ��\\lab2(KNN+NB)\\DATA\\classification_dataset\\test_set.csv",ios::in);
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
				//��֤��ʱ����Ҫ�������� 
				s = line.substr(dot+1,line.size()-dot-1);
				validation.push_back(s);
				break;
			}
		}
		test.push_back(m);
	}
	f1.close();
	
	
	//���ݴ���Ԥ��
//TF����begin*************************************TF*************************************************************
	//����ѵ������TF���� 
	for(int i=0; i<lines.size(); i++){
		double num = 0;//�����ܸ��� 
		for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++)
			num += it->second;
		for(map<string,double>::iterator it=lines[i].begin(); it!=lines[i].end(); it++)
			it->second = it->second / num;		
	}
//TF����end*************************************TF*************************************************************



	//����ÿ����������
	for(int row=0; row<test.size(); row++){		
//TF����begin*************************************TF*************************************************************
		int sizeInWords = 0; //��row�в��������ĵ����� 
		for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++)
			if(words.count(it->first)) //������ֲ��ڵ��ʱ�ĵ��ʣ���������� 
				sizeInWords += it->second;
		for(map<string,double>::iterator it=test[row].begin(); it!=test[row].end(); it++){
			if(words.count(it->first))
				it->second = it->second / sizeInWords; //�������Ӧ��TF�����ֵ
		}
//TF����end*************************************TF*************************************************************
		
		multimap<double,int> distance;//TF����ľ���,ѵ�����ĵڼ������� 		
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
//onehot����������************************************************************************************************ 			
		}


		map<string,int> motion;
		motion["joy"] = 0;
		motion["fear"] = 0;
		motion["sad"] = 0;
		motion["anger"] = 0;
		motion["surprise"] = 0;
		motion["disgust"] = 0;
		
		//�ҵ�ǰk�������������õ���label 
		multimap<double,int>::iterator iter=distance.begin();
		for(int num=0; num<3; num++){ //k��ȡֵ 
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
		if(flag) {//������Ψһ�����ֵʱ��ѡ�þ��������label 
			//ȡ��������������о���������Ǹ�label
			for(multimap<double,int>::iterator it=distance.begin(); it!=distance.end(); it++){
				if(motion[pred] == motion[label[it->second]]){
					predict.push_back(label[it->second]);
					break;
				}
			}
//			predict.push_back(label[distance.begin()->second]);//ȡ��������label 
		}			
		else
			predict.push_back(pred); 	
		
	}
	
	
//���Լ����**********************************************************************	
//	��Ԥ����д���ļ�
	f1.open("E:\\ѧϰ\\������\\�˹�����\\ʵ��\\lab2(KNN+NB)\\DATA\\classification_dataset\\15352010_caiye_KNN_classification.csv",ios::out); 
	f1 << "textid,label\n";
	for(int i=0; i<predict.size();i++) {
		f1 << i+1 << "," << predict[i] << endl;
	}
	f1.close(); 
//���Լ����********************************************************************** 


//��֤�����********************************************************************** 
//	�Աȣ�����׼ȷ�� 
//	int num = 0;
//	for(int i=0; i<predict.size(); i++){
////		cout << validation[i] << " " <<  predict[i] << endl;
//		if(predict[i] == validation[i])
//			num++;
//	} 
//	cout << num*1.0/(predict.size()*1.0) << endl;
//��֤�����************************************************************************ 

	return 0;
}
