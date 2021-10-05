#pragma once
#include "linear_elastic.h"

//#include <iostream>
#include <fstream>
//#include <string.h>
//#include <vector>
using namespace std;

   namespace constants
   {
     // constants have internal linkage by default
     const char cmname[] = "SDCHABOX";
     const int ntens  = 6;
     const int dim    = 3;
     const int nstatv = 29;
     const int nprops = 0;
   }

   extern "C" 
   {
     void param0_sdchabox_(char[],bool*,int);

     void kusdchabox_(double[],double[],double (*ddsdde)[constants::ntens],double*,double*,double*,
		      double*,double[],double[],double*,
     		      double[],double[],double[],double*,double*,double*,double*,double*,char[],
     		      int*,int*,int*,int*,double[],int*,double[constants::dim],double (*drot)[constants::dim],double*,
     		      double*,double (*dfgrd0)[constants::dim],double (*dfgrd1)[constants::dim],int*,int*,int*,int*,int*,int*,
     		      int);

   }

class Umat : public MechanicsLaw
{
public:
     Umat(Constraint c)
        : MechanicsLaw(c)
        , _C(C(1., 0., c))
	, _statevPrev(constants::nstatv)
	, _statevEvaluate(constants::nstatv)
	, _stranPrev(Dim::Q(c))
	, _stressPrev(Dim::Q(c))
    {
      char cmname[strlen(constants::cmname)];
      bool initMat = true;

      // not all constraints are implemented
      // PLANE_STRESS requires another stiffness matrix and must be transfered
      // to UMAT with ntens = 4!!!
      switch(c)
	{
	case PLANE_STRAIN:
	  break;
	case FULL:
	  break;
	default:
	  std::cout << "ERROR: Constraint type "<< c << " is not implemented..."<<std::endl;
	  throw std::exception();
	}
      
      strcpy(cmname, constants::cmname);
      param0_sdchabox_(cmname,&initMat,strlen(cmname));
    }

    void Resize(int n) override
    {
        _statevPrev.Resize(n);
	_statevEvaluate.Resize(n);
	_stranPrev.Resize(n);
	_stressPrev.Resize(n);
    }

    // converts the order of strain/stress components in VectorXd to the order in array for UMAT
  void ConvertVoigtXd2arr(const Eigen::VectorXd& valueXd, double* value)
  {
    const int ntens = _C.rows();
    assert(valueXd.rows() == ntens);

    if (ntens == 6)
      {
	// FULL
  	for ( int j = 0; j != 3; j++ )
  	  {
  	    value[j] = valueXd(j);
  	  }
  	  value[3] = valueXd(5);
  	  value[4] = valueXd(4);
  	  value[5] = valueXd(3);
	  // for ( int j = 0; j != 6; j++ )
  	  // {
	  //   std::cout << j << " " << value[j] << " " << valueXd(j) << std::endl;
  	  // }
      } else if (ntens == 3)
      {
	// PLANE_STRAIN
	value[0] = valueXd(0);  // eps11
	value[1] = valueXd(1);  // eps22 
	value[2] = 0.;          // eps33
	value[3] = valueXd(2);  // 2*eps12 
	value[4] = 0.;          // 2*eps13
	value[5] = 0.;          // 2*eps23

	//       for ( int j = 0; j != 3; j++ )
	// 	  {
	// 	    value[j] = valueXd(j);
	//  	  }
      }
    //   std::cout << " what is value " << value << std::endl;
  }

  // converts the order of strain/stress components in UMAT array to VectorXd in FeniCS
  Eigen::VectorXd ConvertVoigtArr2vectorXd(const double value[], const int sizeOfValue)
  {
    const int ntens = _C.rows();
    assert(sizeOfValue == ntens);
 
    Eigen::VectorXd valueXd(ntens);

     if (ntens == 6)
 	{
	  // FULL
	  for ( int j = 0; j != 3; j++ )
  	    {
  	      valueXd(j) = value[j];
  	    }
  	  valueXd(3) = value[5];
  	  valueXd(4) = value[4];
  	  valueXd(5) = value[3];
  	} else if (ntens == 3)
        {
	  // PLANE_STRAIN
	  valueXd(0) = value[0];  // eps11
	  valueXd(1) = value[1];  // eps22 
	  valueXd(2) = value[3];  // 2*eps12 
	  
	  // for ( int j = 0; j != 3; j++ )
  	  //  {
	  //  valueXd(j) = value[j];
	  //}
	}
     
     return valueXd;
   }

  Eigen::MatrixXd ConvertDdsdde2matrixXd(const double (*ddsdde)[constants::ntens])
  {
        const int ntens = _C.rows();
	assert(constants::ntens < ntens);

	Eigen::MatrixXd ddsddeXd(ntens,ntens);

	if (ntens == 6)
	  {
	    // FULL
	    for ( int i = 0; i != ntens; i++){
	      for ( int j = 0; j != ntens; j++){
		ddsddeXd(i,j) = ddsdde[j][i];     // c++ array = tr(fortran array)
	      }
	    }
	    ddsddeXd.row(3).swap(ddsddeXd.row(5));  // swap 3th and 5th cols and rows
	    ddsddeXd.col(3).swap(ddsddeXd.col(5));  // UMAT to fenics convention 
	  } else if (ntens == 3) {
	  // PLANE_STRAIN
	  ddsddeXd(0,0) = ddsdde[0][0];
	  ddsddeXd(1,1) = ddsdde[1][1];
	  ddsddeXd(2,2) = ddsdde[3][3];
	  
	  ddsddeXd(0,1) = ddsdde[1][0];
	  ddsddeXd(0,2) = ddsdde[3][0];
	  ddsddeXd(1,2) = ddsdde[3][1];
	  
	  ddsddeXd(1,0) = ddsddeXd(0,1);
	  ddsddeXd(2,0) = ddsddeXd(0,2);
	  ddsddeXd(2,1) = ddsddeXd(1,2);

	  // for ( int i = 0; i != ntens; i++){
	  //     for ( int j = 0; j != ntens; j++){
	  // 	ddsddeXd(i,j) = ddsdde[j][i];     // c++ array = tr(fortran array)
	  //     }
	  //   }
	  }
	  
	return ddsddeXd;
  }
  
    void Update(const Eigen::VectorXd& strain, int i) override
    {
      int nstatv = constants::nstatv;
      const int ntens = _C.rows();
	
      Eigen::VectorXd statevXd(nstatv), stressXd(ntens);

      stressXd = Evaluate(strain, i).first;   // the routine overwrites _statevEvaluate
      _statevPrev.Set(_statevEvaluate.Get(i), i);
      _stranPrev.Set(strain, i);
      _stressPrev.Set(stressXd, i);   

      //      std::cout << " OUTPUT FROM RESIZE =================" << std::endl;
      //      std::cout << _stressPrev.Get(i) << std::endl;
      
    }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(const Eigen::VectorXd& strain, int i) override
    {
      // for plain_strain we transform a strain vectors of length 3 to the full vectors of length 6
      // and send to abaqus; therefore for plain_strain and full, the strains and stresses have the
      // length of constant::ntens=6; once we convert the abaqus stress of lemgth 6 to the fenics stress
      // of the length 6 for full or the length 3 for plain_strain, we need to provide the fenics
      // length = _C.rows(), see the return of the function
      int ntens = constants::ntens; //_C.rows();
      int nstatv = constants::nstatv, nprops = constants::nprops;
      int  ndi,nshr,noel,npt,layer,kspt,kstep,kinc;
      double stress[ntens], 
	stran[ntens], dstran[ntens],
	ddsddt[ntens],drplde[ntens];
      double ddsdde[6][6],drot[3][3],dfgrd0[3][3],dfgrd1[3][3];

      double statev[nstatv],props[nprops],time[2],coords[3];

      double sse,spd,scd,pnewdt,rpl,drpldt,dtime,temp,dtemp,predef,dpred,celent;

      temp = 973.15; // 400.;
      // dtime = 0.002;
      // time[0] = 0.024; time[1] = 0.024;
      time[0] = GetTime().first;
      time[1] = GetTime().first;
      dtime = GetTime().second - GetTime().first;

      // get stress, stran and statev from the history variables
      for ( int j = 0; j != nstatv; j++ )
	{
	  statev[j] = _statevPrev.Get(i)(j);
	}
      ConvertVoigtXd2arr(_stranPrev.Get(i), stran);
      ConvertVoigtXd2arr(_stressPrev.Get(i), stress);

      ConvertVoigtXd2arr(strain - _stranPrev.Get(i), dstran);

      char cmname[strlen(constants::cmname)];
//1008      bool initMat = true;
      
      //     assign constants::cmname to cmname;
      //     std::copy(constants::cmname, constants::cmname + strlen(constants::cmname), cmname);
      strcpy(cmname, constants::cmname);
//1008      param0_sdchabox_(cmname,&initMat,strlen(cmname));
      //      std::cout << "... after param0_" << std::endl;


      kusdchabox_(stress,statev,ddsdde,&sse,&spd,&scd,
		  &rpl,ddsddt,drplde,&drpldt,
		  stran,dstran,time,&dtime,&temp,&dtemp,&predef,&dpred,cmname,
		  &ndi,&nshr,&ntens,&nstatv,props,&nprops,coords,drot,&pnewdt,
		  &celent,dfgrd0,dfgrd1,&noel,&npt,&layer,&kspt,&kstep,&kinc,strlen(cmname));
      
      
      Eigen::VectorXd statevXd(nstatv);
      for ( int j = 0; j != nstatv; j++ )
	{
	  statevXd(j) = statev[j];
	}
 
      _statevEvaluate.Set(statevXd, i);
  
      return {ConvertVoigtArr2vectorXd(stress, _C.rows()), ConvertDdsdde2matrixXd(ddsdde)};
    }

private:
    Eigen::MatrixXd _C;

    // history values
    QValues _statevPrev;
    QValues _statevEvaluate;
    QValues _stranPrev;
    QValues _stressPrev; 
};
