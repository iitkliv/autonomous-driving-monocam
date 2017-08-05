#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
using namespace std;

int main(){
  string s1 = "mkdir Filenames";
  system(s1.c_str());
  s1 = "ls tiny-imagenet-200/train/ > folders.txt";
  system(s1.c_str());
  ifstream file_folder("folders.txt");
  string name;
  int i=0;
  while(file_folder.good()){
    getline(file_folder,name);
    if(name.empty())
     	continue;
    i++;
    stringstream num;
    num<<i;
    string s="ls tiny-imagenet-200/train/"+name+"/images/*.JPEG >Filenames/"+num.str()+".txt";
    cout<<s<<endl;
    system(s.c_str());
   }
  return 0;
}
