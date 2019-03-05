#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiApplication.h"
#include "qmessagebox.h"
#include "QDebugStream.h"

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
	QDebugStream* m_qd;

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

private slots:

	void on_radioManual_clicked();
	void on_radioPicture_clicked();
	void on_radioList_clicked();

	void on_pushAddPoint_clicked();
	void on_pushRemovePoint_clicked();
	void on_pushClearInput_clicked();
	
	void on_pushAddSurface_clicked();
	void on_pushAcceptConfig_clicked();
	void on_pushClearConfig_clicked();
	void on_listConfig_currentItemChanged();

	void on_pushCheckData_clicked();
	void on_pushTrace_clicked();
	void on_pushRender_clicked();

	void on_pushShowWavelength_clicked();
	void on_pushDisplayRGB_clicked();
	void on_pushSaveRaw_clicked();
	void on_pushSaveRGB_clicked();

	void on_actionTest_triggered();
};

class listConfigCompanion
{
public:
	void attachTo(QListWidget* widget)
	{
		p_widget = widget;
	}

	void addWavelength(float newwavelength)
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

	void removeWavelength(float wavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		if (token != wavelengths.end())
		{
			wavelengths.erase(token);
			int i = 0;
			for (; i < p_widget->count(); i++)
			{
				if (p_widget->item(i)->text() == QString::number(wavelength))
				{
					break;
				}
			}
			delete p_widget->item(i);
		}
	}

	void removeAllWavelengths()
	{
		p_widget->clear();
		wavelengths.clear();
	}

	void addPoint(float wavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		if (token != wavelengths.end())
		{
			(*token).pointCount++;
		}
	}

	void removePoint(float wavelength)
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

	bool isExist(float wavelength)
	{
		return std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; }) != wavelengths.end();
	}

	bool isEmpty()
	{
		return wavelengths.size() == 0;
	}

	void setHasConfig(float wavelength, bool value)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		if (token != wavelengths.end())
		{
			(*token).hasConfig = value;
		}
	}

	bool getHasConfig(float wavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		if (token != wavelengths.end())
		{
			return (*token).hasConfig;
		}
		return false;
	}

	bool hasWavelength(float wavelength)
	{
		auto token = std::find_if(wavelengths.begin(), wavelengths.end(), [wavelength](wavelengthAndStatus holder) {return holder.wavelength == wavelength; });
		return token != wavelengths.end();
	}

	void getTraceableWavelengths(float*& listTraceable, int& countTraceable, float*& listUntraceable, int& countUntraceable)
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
	QListWidget* p_widget = nullptr;

	struct wavelengthAndStatus
	{
		float wavelength;
		int pointCount = 0;
		bool hasConfig = false;
		bool inputChangedSinceLastProcess = true;

	};
	std::list<wavelengthAndStatus> wavelengths;
};

class tableConfigCompanion
{
public:
	void attachTo(QTableWidget* table)
	{
		p_table = table;
	}

	void checkOutWavelength(float wavelength);

	bool addSurface(float X, float Y, float Z, float diam, float R, float refracI, float asph, int apo);

	bool acceptCurrentConfig();

	bool clearCurrentConfig();

	bool getConfigAt(float wavelength, std::list<tracer::PI_Surface>& output);

private:
	QTableWidget* p_table = nullptr;

	float currentWavelength;
	std::list<tracer::PI_Surface> currentConfig;

	struct wavelengthAndConfig
	{
		float wavelength;
		std::list<tracer::PI_Surface> surfaces;
	};

	std::list<wavelengthAndConfig> configs;
};