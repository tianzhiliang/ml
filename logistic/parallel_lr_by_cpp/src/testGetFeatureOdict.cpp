#include <logisticModel.h>

int main(int argc,char** argv){
	if(3!=argc){
		fprintf(stderr,"ERROR,%s confPath confFile\n",argv[0]);
		return -1;
	}

	string confPath;
	string confFile;

	confPath=string(argv[1]);
	confFile=string(argv[2]);	

	MPIC::Obj().initMPI(argc,argv);

	fprintf(stderr,"NOTICE,argv: confPath:%s confFile:%s\n",argv[1],argv[2]);
	getFeatureOdict(confPath,confFile);

	MPIC::Obj().freeMPI();	
	
	return 0;
}
