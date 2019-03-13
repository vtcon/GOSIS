#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiApplication.h"
#include <qthread.h>

#include "qmessagebox.h"
#include "QDebugStream.h"
#include "ConsoleOut.h"

#include "../test2/ProgramInterface.h"

#include <list>

class QtGuiApplication : public QMainWindow
{
	Q_OBJECT

public:
	QtGuiApplication(QWidget *parent = Q_NULLPTR);

	struct inputTableItem
	{
		int uniqueID;
		tracer::PI_LuminousPoint point;
	};
	std::list<inputTableItem> inputTableData;

private:
	Ui::QtGuiApplicationClass ui;

	//output for std::cout
	ConsoleOut* w2 = nullptr;
	QThread consolethread;


	//state machine
	unsigned short int stateCounter = 0;
	//0: new Data
	//1: Data Checked-in
	//2: Traced
	//3: Rendered == state 0
	void stateNext()
	{
		stateCounter += 1;
		if (stateCounter > 3)
		{
			//undefined behavior
			__debugbreak();
		}
		state();
	}

	void stateReset()
	{
		stateCounter = 0;
		state();
	}

	void state()
	{
		switch (stateCounter)
		{
		case 0:
			ui.pushTrace->setEnabled(false);
			ui.pushRender->setEnabled(false);
			ui.pushShowWavelength->setEnabled(false);
			ui.pushDisplayRGB->setEnabled(false);
			ui.pushSaveRaw->setEnabled(false);
			ui.pushSaveRGB->setEnabled(false);
			ui.progressTrace->setValue(0);
			ui.progressRender->setValue(0);
			break;
		case 1:
			ui.pushTrace->setEnabled(true);
			ui.pushRender->setEnabled(false);
			ui.pushShowWavelength->setEnabled(false);
			ui.pushDisplayRGB->setEnabled(false);
			ui.pushSaveRaw->setEnabled(false);
			ui.pushSaveRGB->setEnabled(false);
			break;
		case 2:
			ui.pushTrace->setEnabled(false);
			ui.pushRender->setEnabled(true);
			ui.pushShowWavelength->setEnabled(false);
			ui.pushDisplayRGB->setEnabled(false);
			ui.pushSaveRaw->setEnabled(false);
			ui.pushSaveRGB->setEnabled(false);
			break;
		case 3:
			ui.pushTrace->setEnabled(false);
			ui.pushRender->setEnabled(false);
			ui.pushShowWavelength->setEnabled(true);
			ui.pushDisplayRGB->setEnabled(true);
			ui.pushSaveRaw->setEnabled(true);
			ui.pushSaveRGB->setEnabled(true);
			break;
		default:
			//undefined behavior
			break;
		}
	}

public slots:
	void updateRenderProgressBar(int newvalue);
	void updateRenderProgressDirectly();
	void timerTest();
	void updateResUsage();

private slots:

	void on_radioManual_clicked();
	void on_radioPicture_clicked();
	void on_radioList_clicked();

	void on_pushAddPoint_clicked();
	void on_pushRemovePoint_clicked();
	void on_pushClearInput_clicked();
	
	void on_listConfig_currentItemChanged();

	void on_pushAddSurface_clicked();
	void on_pushModifySurface_clicked();
	void on_pushRemoveSurface_clicked();

	void on_pushAcceptConfig_clicked();
	void on_pushClearConfig_clicked();
	void on_pushCloneConfig_clicked();
	void on_pushSaveConfig_clicked();
	void on_pushLoadConfig_clicked();

	void on_pushCheckData_clicked();
	void on_pushTrace_clicked();
	void on_pushRender_clicked();

	void on_pushShowWavelength_clicked();
	void on_pushDisplayRGB_clicked();
	void on_pushSaveRaw_clicked();
	void on_pushSaveRGB_clicked();

	void on_actionTest_triggered();
	void on_actionConsoleOut_triggered();
	void on_actionPreferences_triggered();

	void on_pushSelectImage_clicked();
	void on_pushClearImage_clicked();
};

class listConfigCompanion
{
public:
	static void attachTo(QListWidget* widget)
	{
		p_widget = widget;
	}

	static void addWavelength(float newwavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [newwavelength](wavelengthAndStatus holder) {return holder.wavelength == newwavelength; });
		if (token == wavelengths.end())
		{
			wavelengthAndStatus newholder;
			newholder.wavelength = newwavelength;
			wavelengths.push_back(newholder);
			new QListWidgetItem(QString::number(newwavelength), p_widget);
		}
	}

	static void removeWavelength(float wavelength);

	static void removeAllWavelengths();
	
	static void addPoint(float wavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		if (token != wavelengths.end())
		{
			(*token).pointCount++;
		}
	}

	static void removePoint(float wavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		if (token != wavelengths.end())
		{
			(*token).pointCount--;
			if ((*token).pointCount <= 0)
			{
				removeWavelength(wavelength);
			}
		}
	}

	static bool isExist(float wavelength)
	{
		return std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; }) != wavelengths.end();
	}

	static bool isEmpty()
	{
		return wavelengths.size() == 0;
	}

	static void setHasConfig(float wavelength, bool value)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		if (token != wavelengths.end())
		{
			(*token).hasConfig = value;
		}
	}

	static bool getHasConfig(float wavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		if (token != wavelengths.end())
		{
			return (*token).hasConfig;
		}
		return false;
	}

	static bool hasWavelength(float wavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		return token != wavelengths.end();
	}

	static void getTraceableWavelengths(float*& listTraceable, int& countTraceable, float*& listUntraceable, int& countUntraceable)
	{
		std::vector<float> templist;
		std::vector<float> templistNon;
		for (auto eachwavelength : wavelengths)
		{
			if (eachwavelength.hasConfig == true)
			{
				templist.push_back(eachwavelength.wavelength);
			}
			else
			{
				templistNon.push_back(eachwavelength.wavelength);
			}
		}

		if ((countTraceable = templist.size()) != 0)
		{
			listTraceable = new float[countTraceable];
			int i = 0;
			for (auto eachwavelength : templist)
			{
				listTraceable[i] = eachwavelength;
				i++;
			}
		}

		if ((countUntraceable = templistNon.size()) != 0)
		{
			listUntraceable = new float[countUntraceable];
			int i = 0;
			for (auto eachwavelength : templistNon)
			{
				listUntraceable[i] = eachwavelength;
				i++;
			}
		}
	}

private:
	static QListWidget* p_widget;

	struct wavelengthAndStatus
	{
		float wavelength;
		int pointCount = 0;
		bool hasConfig = false;
		bool inputChangedSinceLastProcess = true;

	};
	static std::list<wavelengthAndStatus> wavelengths;
};

class tableConfigCompanion
{
public:
	static void attachTo(QTableWidget* table)
	{
		p_table = table;
	}

	static void checkOutWavelength(float wavelength);

	static bool addSurface(float X, float Y, float Z, float diam, float R, float refracI, float asph, int apo, const char* apoPath = "");

	static bool modifySurfaceAtCurrentRow();

	static bool deleteSurfaceAtCurrentRow();

	static bool acceptCurrentConfig();

	static bool clearCurrentConfig();

	static bool cloneToCurrentConfig();

	static bool saveCurrentConfig(QString path);

	static bool loadToCurrentConfig(QString path);

	static bool clearConfigAt(float wavelength);

	static bool getConfigAt(float wavelength, std::list<tracer::PI_Surface>& output);

	static void clearApoPathList();

	static void clearAllData();

private:
	static QTableWidget* p_table;

	static float currentWavelength;
	static std::list<tracer::PI_Surface> currentConfig;

	struct wavelengthAndConfig
	{
		float wavelength;
		std::list<tracer::PI_Surface> surfaces;
	};

	static std::list<wavelengthAndConfig> configs;

	static std::list<QByteArray> apoPathList; //save the paths in Latin1 encoding
};
