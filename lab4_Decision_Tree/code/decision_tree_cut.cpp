#include<bits/stdc++.h>
using namespace std;

struct node{
	string leaf = "0";//标记叶节点和分类 
	int attr = 0;
	string _class = "*";
	vector<node*> children;
	vector< vector<string> > data;
};
vector< vector<string> > all;// 全部数据
vector< vector<string> > train;//训练集的数据，第0位为label
vector< vector<string> >  validate;//验证集 
vector< vector<string> > test;//测试集 
vector<int> attr;//特征集 
fstream f1,f2;

vector< vector<string> > readData(string address);
bool meet_with_bound(node *p);
int choose_attr(vector<vector<string>> data, int i);
vector<vector<vector<string>>> divide_data(vector<vector<string>> D, int chosen);
void recursive(node *p,int medthod);
void printTree(node *p);
pair<int,int> cutLeaf(node *p);

int main(){
	
	
	all = readData("E:\\学习\\大三上\\人工智能\\实验\\lab4_Decision_Tree\\YStrain1.csv");
	test = readData("E:\\学习\\大三上\\人工智能\\实验\\lab4_Decision_Tree\\YStest1.csv"); 
	for(int i=1; i<all[0].size(); i++)
		attr.push_back(i);

	
//	f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab4_Decision_Tree\\res.csv",ios::out);
//	for(int t=0; t<300; t++){
//		int n = t+1;
//		string sum = "";
//		while(n) {
//			char c = n%10 + '0';
//			sum = c + sum;
//			n /= 10;
//		}	
//		f1.open("E:\\学习\\大三上\\人工智能\\实验\\lab4_Decision_Tree\\res\\res"+sum+".csv",ios::out);
		
		
//		srand(time(0));
//		random_shuffle(all.begin(),all.end());
//		train.assign(all.begin(),all.begin()+all.size()*0.75); 
//		validate.assign(all.begin()+all.size()*0.75,all.end());
		
//		train.assign(all.begin(),all.begin()+t); 
//		validate.assign(all.begin()+t,all.begin()+t+150);
//		train.insert(train.end(),all.begin()+t+150,all.begin()+324+t);
//		validate.insert(validate.end(),all.begin()+324+t,all.begin()+t+474);
//		train.insert(train.end(),all.begin()+474+t,all.end());
		
		for(int method=1; method<2; method++){	
			node *root = new node;
			root->data = all;
			recursive(root,method);//建树 
//			cutLeaf(root);//剪枝
			 
			printTree(root);
			 
			//测试集预测 
			vector<string> pred;
			for(int i=0; i<test.size(); i++){
				node *p = root;
				while(p->leaf == "0"){
					bool appear = 0;
					for(int j=0; j<p->children.size(); j++){
						if(p->children[j]->_class == test[i][p->attr]){
							p = p->children[j];
							appear = 1;
							break;
						}
					}
					if(!appear)
						p = p->children[0];
				}
				pred.push_back(p->leaf);
				cout << p->leaf << endl;
//				f1 << p->leaf << endl; 
			}
			//验证集预测 
//			vector<string> predict;
//			for(int i=0; i<validate.size(); i++){
//				node *p = root;
//				while(p->leaf == "0"){
//					bool appear = 0;
//					for(int j=0; j<p->children.size(); j++){
//						if(p->children[j]->_class == validate[i][p->attr]){
//							p = p->children[j];
//							appear = 1;
//							break;
//						}
//					}
//					if(!appear)
//						p = p->children[0];
//				}
//				predict.push_back(p->leaf);
//			}
//					
//			int cnt = 0;
//			for(int i=0; i<predict.size(); i++){
//				if(predict[i] == validate[i][0])
//					cnt++;
//			}
//			f2 << 1.0*cnt/predict.size()*1.0 << ",";
			
		}
//		f2 << endl;
		
//		f1.close();	
//	}	
//	f2.close();
	return 0;
}

void printTree(node *p){
	
	if(p->children.size() == 0){
		cout << "*********" << p->leaf << " " << p->_class << endl;
		return ;
	}	
	cout << "#";
	cout << p->leaf << " " << p->_class << "  " <<  p->attr << endl;
	for(int i=0; i<p->children.size(); i++)
		printTree(p->children[i]);
}


//剪枝 
//返回的是当前子树的错误个数和叶子节点数 
pair<int,int> cutLeaf(node *p){
	if(p->leaf != "0"){
		return make_pair(0,1);
	}
	//剪枝后误判个数 
	double positive=0, negative=0, errorAfter=0, errorBefore=0, leave=0; 
	for(int i=0; i<p->data.size(); i++){
		if(p->data[i][0] == "1")	positive+=1;
	} 
	negative = p->data.size()-positive;
	errorAfter = (positive >= negative) ? negative : positive;
	//没剪枝时的误判个数和叶子节点 
	for(int i=0; i<p->children.size(); i++){
		if(p->children[i]->leaf == "0"){//非叶子节点
			pair<int,int> temp = cutLeaf(p->children[i]);
			errorBefore += temp.first;
			leave += temp.second;
		}
		else{//叶子节点
			leave += 1; 
			for(int j=0; j<p->data.size(); j++){
				if(p->data[j][p->attr] == p->children[i]->_class)//样本属于该叶子节点的分类 
					if(p->data[j][0] != p->children[i]->leaf)
						errorBefore += 1;
			}
		} 
	}
	if((errorAfter+0.5) < (errorBefore+0.5*leave)){//剪枝后错误率小 
		p->leaf = (positive > negative) ? "1" : "-1";
		return make_pair(0,1);
	} 
	else//不剪枝 
		return make_pair(errorBefore,leave);
} 


//递归建树 
void recursive(node *p,int method)
{
	if(meet_with_bound(p)){ return; } //边界条件 
	int attr_chosen = choose_attr(p->data,method);//method= 1:ID3：2:C4.5：3:CART
	p->attr = attr_chosen; //选中的特征 
	vector<vector<vector<string>>> subsets = divide_data(p->data, attr_chosen);//分割数据集 
	//删除选中的特征 
	vector<int>::iterator it = attr.begin();
	while((*it)!= attr_chosen)	it++;
	attr.erase(it);
	//将每一个分类添加为孩子节点
	for(int i=0; i<subsets.size(); i++){ 
		node *subnode = new node;
		subnode->data = subsets[i];
		p->children.push_back(subnode);
		subnode->_class = subsets[i][0][attr_chosen];
		//对孩子节点进行递归建树 
		recursive(subnode,method);
	}
	//添加回前面被删掉的特征 
	it = attr.begin();
	while((*it) < attr_chosen)	it++;
	attr.insert(it,attr_chosen);
	return;
}


//边界条件 
bool meet_with_bound(node *p)
{	//所有样本属于同一类别
	bool same = 1;
	for(int i=1; i<p->data.size(); i++){
		if(p->data[i][0] != p->data[i-1][0]){
			same = 0;
			break;
		}
	}
	if(same){
		p->leaf = p->data[0][0];
		return 1;
	}
	//在特征集中所有特征上取值相同，或者特征集为空，无法再分，取出现次数多的作为label 
	same = 1;
	for(int i=0; i<attr.size(); i++){
		for(int j=1; j<p->data.size(); j++){
			if(p->data[j][attr[i]] != p->data[j-1][attr[i]]){
				same = 0;
				break;
			}
		}
		if(!same)	break;
	}
	if(attr.size()==0 || same){
		int posi = 0;
		for(int i=0; i<p->data.size(); i++){
			if(p->data[i][0] == "1")
				posi++;
		} 
		if(posi >= (p->data.size()-posi))
			p->leaf = "1";
		else
			p->leaf = "-1";
		return 1;
	}
	return 0;
}



//三种特征选择的方法 
int ID_3(vector<vector<string>> data){
	double size = data.size() * 1.0;
	//计算经验熵 H(D)
	double positive = 0;
	for(int i=0; i<data.size(); i++)
		if(data[i][0] == "1")	positive += 1;
	double negative = size - positive;
	double H_D = -(positive/size)*(log(positive/size)/log(2)) - (negative/size)*(log(negative/size)/log(2));
	//计算每个特征下的条件熵			 
	vector<double> gain;//信息增益 
	for(int i=0; i<attr.size(); i++){  //遍历每个属性 
		double res = 0;
		map<string,bool> visited;
		for(int j=0; j<data.size(); j++){
			if(visited.count(data[j][attr[i]]))	continue;//在特征attr[i]下的该分类已经被计算过 
			visited[data[j][attr[i]]] = true;
			
			double cnt=0, posi=0, nega=0;
			for(int k=j; k<data.size(); k++){//找出所有和样本j在特征attr[i]下分类一样的样本 
				if(data[j][attr[i]] == data[k][attr[i]]){
					cnt += 1;
					if(data[k][0] == "1")	posi +=1;
					else if(data[k][0]=="-1")	nega += 1;
				}
			}
			if(posi!=0 && nega!=0)//为了避免log（0）出错 
				res += (cnt/size) * (-(posi/cnt)*(log(posi/cnt)/log(2)) - (nega/cnt)*(log(nega/cnt)/log(2))); 
		}
		gain.push_back(H_D - res);
	}
	int max = 0; 
	for(int i=0; i<gain.size(); i++){
		cout << "特征" << attr[i] << "的信息增益：" << gain[i] << endl;
		if(gain[i] > gain[max])
			max = i;
	}
	cout << "选择第" << attr[max]	<< "个特征作为节点\n"; 
	return attr[max];
}

int C4_5(vector<vector<string>> data){
	double size = data.size() * 1.0;
	//计算经验熵H(D)
	double positive = 0;
	for(int i=0; i<data.size(); i++)
		if(data[i][0] == "1")	positive += 1;
	double negative = size - positive;
	double H_D = -(positive/size)*(log(positive/size)/log(2)) - (negative/size)*(log(negative/size)/log(2));
	//计算每个特征下的条件熵			 
	vector<double> gain;//信息增益率 
	for(int i=0; i<attr.size(); i++){ //遍历每个属性
		double res = 0;
		double spiltinfo = 0;
		map<string,bool> visited;
		for(int j=0; j<data.size(); j++){
			if(visited.count(data[j][attr[i]]))	continue;
			visited[data[j][attr[i]]] = true;
			
			double cnt=0, posi=0, nega=0;
			for(int k=j; k<data.size(); k++){
				if(data[j][attr[i]] == data[k][attr[i]]){
					cnt += 1;
					if(data[k][0] == "1")	posi +=1;
					else if(data[k][0]=="-1")	nega += 1;
				}
			}
			if(posi!=0 && nega!=0)
				res += (cnt/size) * (-(posi/cnt)*(log(posi/cnt)/log(2)) - (nega/cnt)*(log(nega/cnt)/log(2))); 
			spiltinfo += -(cnt/size)*(log(cnt/size)/log(2));//比ID3多的唯一一步 
				
		}
		gain.push_back((H_D-res)/spiltinfo);
	}
	int max = 0; 
	for(int i=0; i<gain.size(); i++)
		if(gain[i] > gain[max])
			max = i;
	return attr[max];
}

int CART_gini(vector<vector<string>> data){
	vector<double> index;
	//计算每种特征的基尼系数 
	for(int i=0; i<attr.size(); i++){ //遍历每个属性
		double res = 0;
		map<string,bool> visited;
		for(int j=0; j<data.size(); j++){//遍历数据集 
			if(visited.count(data[j][attr[i]]))	continue;//在特征attr[i]下的该分类已经被计算过 
			visited[data[j][attr[i]]] = true;

			double cnt=0, positive=0, negative=0;
			for(int k=j; k<data.size(); k++){//找出数据集中所有和样本j在特征attr[i]下分类一样的样本 
				if(data[j][attr[i]] == data[k][attr[i]]){
					cnt += 1;
					if(data[k][0] == "1")	positive += 1;
					else if(data[k][0] == "-1")	negative += 1;
				}
			}
			//p(a)*[1-sum(pi^2)]
			res += (cnt/data.size()*1.0) * (1-pow((positive/cnt),2)-pow((negative/cnt),2));
		} 
		index.push_back(res);
	}
	int min = 0;
	for(int i=1; i<index.size(); i++){
		if(index[i] < index[min])
			min = i;
	}
	return attr[min];
}
//选择决策点 
int choose_attr(vector<vector<string>> data, int i)
{
	if(i == 1){
		return ID_3(data);
	} else if(i == 2){
		return C4_5(data);
	} else if(i == 3){
		return CART_gini(data);
	} 
}


//数据集分割 
vector<vector<vector<string>>> divide_data(vector<vector<string>> D, int chosen)
{
	vector<vector<vector<string>>> v;
	map<string,int> visited;
	for(int i=0; i<D.size(); i++){
		if(visited.count(D[i][chosen]))	continue;//该分类已经被计算过 
		visited[D[i][chosen]] = 1;
		vector<vector<string>> temp;
		for(int k=i; k<D.size(); k++){
			if(D[k][chosen] == D[i][chosen]){//在特征chosen下的分类相同 
				vector<string> line;
				line = D[k];		
				temp.push_back(line);
			}		
		}	
		v.push_back(temp);
	}
	return v;
}

//读数据 
vector< vector<string> > readData(string address){
	vector< vector<string> > data;
	f1.open(address,ios::in);
	string s,num;
	int cnt = 0;
	while(getline(f1,s)){
		cnt++;
		int begin = 0;
		int dot = s.find(",",begin);
		vector<string> line;
		while(1){
			if(dot == -1)
				break; 
			num = s.substr(begin,dot-begin);
			line.push_back(num);
				
			begin = dot + 1;
			dot = s.find(",",begin);	
		}
		line.insert(line.begin(),s.substr(begin,dot-begin));//label在第0位 
		data.push_back(line);
	}
	f1.close();
	return data;
}
