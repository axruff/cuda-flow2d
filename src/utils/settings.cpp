/**
* @file    2D Optical flow using NVIDIA CUDA
* @author  Institute for Photon Science and Synchrotron Radiation, Karlsruhe Institute of Technology
*
* @date    2015-2018
* @version 0.5.0
*
*
* @section LICENSE
*
* This program is copyrighted by the author and Institute for Photon Science and Synchrotron Radiation,
* Karlsruhe Institute of Technology, Karlsruhe, Germany;
*
* The current implemetation contains the following licenses:
*
* 1. TinyXml package:
*      Original code (2.0 and earlier )copyright (c) 2000-2006 Lee Thomason (www.grinninglizard.com). <www.sourceforge.net/projects/tinyxml>.
*      See src/utils/tinyxml.h for details.
*
*/

#include "settings.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace OpticFlow;




void Settings::LoadSettingsManually()
{

	this->sigma = 1.2;

	// Solver settings
	this->iterInner = 5;
	this->iterOuter = 20;

	this->alpha = 40;

	this->levels = 1;
	this->warpScale = 0.5;


	this->e_smooth = 0.1;
	this->e_data = 0.001;

}

int Settings::LoadSettings(string fileName)
{

	string m_name;

	TiXmlDocument doc(fileName.c_str());
	
	if (!doc.LoadFile()) {
		cout<<"Cannot read settings file: "<<fileName<<endl;
		return -1;
	}

	//Read file the whole settings file content

	std::ifstream f(fileName.c_str());
	std::stringstream buffer;
	buffer << f.rdbuf();
	content = buffer.str();


	TiXmlHandle hDoc(&doc);
	TiXmlElement* pElem;
	TiXmlHandle hRoot(0);

	
	pElem=hDoc.FirstChildElement().Element();

	if (!pElem) {
		cout<<"Problem with parsing settings file: "<<fileName<<endl;
		return -1;
	}
	m_name=pElem->Value();

	// save this for later
	hRoot=TiXmlHandle(pElem);

    int value;
    float dbValue;
	

	pElem=hRoot.FirstChild( "Input" ).FirstChild("Path").Element();
	this->inputPath = pElem->Attribute("inputPath");

	pElem=hRoot.FirstChild( "Output").FirstChild("Path").Element();
	this->outputPath = pElem->Attribute("outputPath");

	
	this->fileName1 = hRoot.FirstChild( "Input" ).FirstChild("Mode").FirstChild("Files").Element()->Attribute("file1");
	this->fileName2 = hRoot.FirstChild( "Input" ).FirstChild("Mode").FirstChild("Files").Element()->Attribute("file2");
	

	hRoot.FirstChild( "Parameters" ).FirstChild("Method").Element()->QueryIntAttribute("key", &value);
	this->press_key = value;

	hRoot.FirstChild( "Parameters" ).FirstChild("Scheme").FirstChild("Presmooth").Element()->QueryFloatAttribute("sigma", &dbValue);
	this->sigma = dbValue;


	hRoot.FirstChild( "Parameters" ).FirstChild("Solver").FirstChild("Iterations").Element()->QueryIntAttribute("inner", &value);
	this->iterInner = value;
	hRoot.FirstChild( "Parameters" ).FirstChild("Solver").FirstChild("Iterations").Element()->QueryIntAttribute("outer", &value);
	this->iterOuter = value;

	hRoot.FirstChild( "Parameters" ).FirstChild("Solver").FirstChild("Warping").Element()->QueryIntAttribute("levels", &value);
	this->levels= value;

	hRoot.FirstChild( "Parameters" ).FirstChild("Solver").FirstChild("Warping").Element()->QueryFloatAttribute("scaling", &dbValue);
	this->warpScale= dbValue;

	hRoot.FirstChild( "Parameters" ).FirstChild("Solver").FirstChild("Warping").Element()->QueryIntAttribute("medianRadius", &value);
	this->medianRadius = value;


	hRoot.FirstChild( "Parameters" ).FirstChild("Solver").FirstChild("Model").Element()->QueryFloatAttribute("alpha", &dbValue);
	this->alpha= dbValue;



	hRoot.FirstChild( "Parameters" ).FirstChild("Solver").FirstChild("Model").Element()->QueryFloatAttribute("e_smooth", &dbValue);
	this->e_smooth= dbValue;
	hRoot.FirstChild( "Parameters" ).FirstChild("Solver").FirstChild("Model").Element()->QueryFloatAttribute("e_data", &dbValue);
	this->e_data= dbValue;




	

	return 0;


}




