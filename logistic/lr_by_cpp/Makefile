#COMAKE2 edit-mode: -*- Makefile -*-
####################64Bit Mode####################
ifeq ($(shell uname -m),x86_64)
CC=gcc
CXX=g++
CXXFLAGS=-g \
  -pipe \
  -W \
  -Wall \
  -fPIC
CFLAGS=-g \
  -pipe \
  -W \
  -Wall \
  -fPIC
CPPFLAGS=-D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\"
INCPATH=-I. \
  -I./include \
  -I./output \
  -I./output/include
DEP_INCPATH=

#============ CCP vars ============
CCHECK=@ccheck.py
CCHECK_FLAGS=
PCLINT=@pclint
PCLINT_FLAGS=
CCP=@ccp.py
CCP_FLAGS=


#COMAKE UUID
COMAKE_MD5=78085bdbdc1a831058b80b1dff038908  COMAKE


.PHONY:all
all:comake2_makefile_check liblogisticModel.a LRTrainTool LRTestTool rm1 
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mall[0m']"
	@echo "make all done"

.PHONY:comake2_makefile_check
comake2_makefile_check:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mcomake2_makefile_check[0m']"
	#in case of error, update 'Makefile' by 'comake2'
	@echo "$(COMAKE_MD5)">comake2.md5
	@md5sum -c --status comake2.md5
	@rm -f comake2.md5

.PHONY:ccpclean
ccpclean:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mccpclean[0m']"
	@echo "make ccpclean done"

.PHONY:clean
clean:ccpclean
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mclean[0m']"
	rm -rf liblogisticModel.a
	rm -rf ./output/lib/liblogisticModel.a
	rm -rf ./output/include/logisticModel.h
	rm -rf LRTrainTool
	rm -rf ./output/bin/LRTrainTool
	rm -rf LRTestTool
	rm -rf ./output/bin/LRTestTool
	rm -rf rm1
	rm -rf logisticModel_logisticModel.o
	rm -rf LRTrainTool_testLRTrain.o
	rm -rf LRTestTool_testLRPred.o

.PHONY:dist
dist:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mdist[0m']"
	tar czvf output.tar.gz output
	@echo "make dist done"

.PHONY:distclean
distclean:clean
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mdistclean[0m']"
	rm -f output.tar.gz
	@echo "make distclean done"

.PHONY:love
love:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mlove[0m']"
	@echo "make love done"

liblogisticModel.a:logisticModel_logisticModel.o \
  logisticModel.h
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mliblogisticModel.a[0m']"
	ar crs liblogisticModel.a logisticModel_logisticModel.o
	mkdir -p ./output/lib
	cp -f --link liblogisticModel.a ./output/lib
	mkdir -p ./output/include
	cp -f --link logisticModel.h ./output/include

LRTrainTool:LRTrainTool_testLRTrain.o \
  liblogisticModel.a
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mLRTrainTool[0m']"
	$(CXX) LRTrainTool_testLRTrain.o -Xlinker "-(" liblogisticModel.a  -lpthread \
  -lcrypto \
  -lrt -Xlinker "-)" -o LRTrainTool
	mkdir -p ./output/bin
	cp -f --link LRTrainTool ./output/bin

LRTestTool:LRTestTool_testLRPred.o \
  liblogisticModel.a
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mLRTestTool[0m']"
	$(CXX) LRTestTool_testLRPred.o -Xlinker "-(" liblogisticModel.a  -lpthread \
  -lcrypto \
  -lrt -Xlinker "-)" -o LRTestTool
	mkdir -p ./output/bin
	cp -f --link LRTestTool ./output/bin

rm1:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mrm1[0m']"
	rm -f *.o *.~

logisticModel_logisticModel.o:logisticModel.cpp \
  logisticModel.h
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mlogisticModel_logisticModel.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) $(CPPFLAGS) $(CXXFLAGS)  -o logisticModel_logisticModel.o logisticModel.cpp

LRTrainTool_testLRTrain.o:testLRTrain.cpp \
  logisticModel.h
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mLRTrainTool_testLRTrain.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) $(CPPFLAGS) $(CXXFLAGS)  -o LRTrainTool_testLRTrain.o testLRTrain.cpp

LRTestTool_testLRPred.o:testLRPred.cpp \
  logisticModel.h
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mLRTestTool_testLRPred.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) $(CPPFLAGS) $(CXXFLAGS)  -o LRTestTool_testLRPred.o testLRPred.cpp

endif #ifeq ($(shell uname -m),x86_64)


