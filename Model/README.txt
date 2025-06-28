#include<iostream>
#include<vector>
#include<unordered_map>
using namespace std;

int main(){
    int n;
    cin>>n;
    
    vector<int>arr(n);
    for(int i=0; i<n;i++){
        cin>>arr[i];
    }
    unordered_map<int,int>freq;
    for(int i=0;i<n;i++){
        freq[arr[i]]++;
    }
    
    int maxDistance=0;
    for(auto&p : freq){
        if(p.second%2==1){
            int element=p.first;
            
            int first=-1;
            for(int i=0;i<n;i++){
                if(arr[i]==element){
                    first=i;
                    break;
                }
            }
            int last=-1;
            for(int i=n-1;i>=0;i--){
                if(arr[i]==element){
                    last=i;
                    break;
                }
            }
            int distance= abs(last-first);
            maxDistance= max(maxDistance,distance);
        }
    }
    cout<<maxDistance<<endl;
    return 0;