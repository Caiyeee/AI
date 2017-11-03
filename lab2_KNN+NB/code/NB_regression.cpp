#include<bits/stdc++.h>
using namespace std;

//0:anger, 1:disgust, 2:fear, 3:joy, 4:sad, 5:surprise
vector< map<string,int> > lines;
vector<int> length;//每一行的长度
map<string,int> wordlist;
vector< vector<double> > label;//行：某个文章的六种情绪
vector< vector<double> > predict;//预测值 
fstream f1,f2;


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
	//读取训练集的数据 
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\regression_dataset\\train_set.csv",ios::in);
	string line,s;
	getline(f1,line);
	
	while(getline(f1,line)) {
		map<string,int> wordsOfLine;
		int dot = line.find(",");//逗号的位置 
		int begin = 0;
		int end = line.find(" ",begin);//空格的位置 

		int len = 0;
		while(1) {
			if(end != -1)
				s = line.substr(begin,end-begin);
			else
				s = line.substr(begin,dot-begin);
			
			if(!wordlist.count(s))
				wordlist[s] = 1;
			if(!wordsOfLine.count(s))
				wordsOfLine[s] = 1;
			else
				wordsOfLine[s] += 1;
			len++;
			
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
		length.push_back(len);
		
		
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
	
	
	//读取验证集/测试集的数据
	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\regression_dataset\\test_set.csv",ios::in);
//	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\regression_dataset\\validation_set.csv",ios::in);
	f2.open("E:\\学习\\大三上\\人工智能\\实验\\lab2(KNN+NB)\\DATA\\regression_dataset\\15352010_caiye_NB_regression.csv",ios::out); 
	f2 << "textid,anger,disgust,fear,joy,sad,surprise\n";
	getline(f1,line);
	int textid = 0;
	while(getline(f1,line)) {
		int begin = 0;
		int end = line.find(" ",begin);
		int dot = line.find(",",begin);
		//读取该样例的所有单词 
		map<string,int> wordsOfLine;
		while(1) {
			if(end != -1)
				s = line.substr(begin,end-begin);
			else
				s = line.substr(begin,dot-begin);
			
			if(!wordsOfLine.count(s))
				wordsOfLine[s] = 1;
			else
				wordsOfLine[s] += 1;
				
			if(end != -1) {
				begin = end + 1;
				end = line.find(" ",begin);
			}
			else
				break;
		}	
		
		vector<double> motion;
		for(int i=0; i<6; i++){//六种情绪 
			double motion_label = 0;
			for(int j=0; j<lines.size(); j++){//每一个训练集的样例 
				double motion_line = 1;
				for(map<string,int>::iterator it=wordsOfLine.begin(); it!=wordsOfLine.end(); it++){
					double temp = 0;//Xk，单词在样本中出现的次数 
					if(lines[j].count(it->first))
						temp += lines[j][it->first];
					temp += 0.01;
					//p(Xk|dj,ei) = (Xk+0.01) / (sum(Xk)+0.01*K) 
					motion_line *= temp / (1.0*length[j]+0.01*wordlist.size());
				}
				motion_line *= label[j][i];
				motion_label += motion_line;
			}
			motion.push_back(motion_label);
		}
		double sum = 0;
		for(int i=0; i<6; i++)
			sum += motion[i];
			
		f2 << ++textid;
		for(int i=0; i<6; i++) {
			motion[i] = motion[i] / sum;
			f2 << "," << fixed << setprecision(6) << motion[i];
		}
		f2 << endl;
			
		
	}
	f1.close(); 
	f2.close();
	
	
	return 0;
} 
