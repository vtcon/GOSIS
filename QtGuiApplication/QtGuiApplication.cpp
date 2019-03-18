#include "QtGuiApplication.h"

#include "AddPointDialog.h"
#include "AddSurfaceDialog.h"
#include "CloneConfigDialog.h"
#include "PreferenceDialog.h"

#include <qobject.h>

#include "qmessagebox.h"
#include <QFileDialog>
#include <qtimer.h>

#include <QTextStream>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <thread>

#include "windows.h"
#include "psapi.h"

//supporting structures
#include "RenderProgressEmittor.h"

//static initialization
QListWidget* listConfigCompanion::p_widget = nullptr;
std::list<listConfigCompanion::wavelengthAndStatus> listConfigCompanion::wavelengths;
QTableWidget* tableConfigCompanion::p_table = nullptr;
float tableConfigCompanion::currentWavelength;
std::list<tracer::PI_Surface> tableConfigCompanion::currentConfig;
std::list<tableConfigCompanion::wavelengthAndConfig> tableConfigCompanion::configs;
std::list<QByteArray> tableConfigCompanion::apoPathList;

//global variables
//std::vector<float> inputWavelengths;
//listConfigCompanion wavelengthList;
//tableConfigCompanion configTable;
std::vector<int> outputImageIDs;

QtGuiApplication::QtGuiApplication(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	ui.tableInput->setColumnHidden(3, true);

	//
	//m_qd = new QDebugStream(std::cout, ui.textEditProcess); //Redirect Console output to QTextEdit
	//m_qd->QDebugStream::registerQDebugMessageHandler(); //Redirect qDebug() output to QTextEdit


	//attach global variables to widgets
	listConfigCompanion::attachTo(ui.listConfig);
	tableConfigCompanion::attachTo(ui.tableConfig);

	//start the state machine
	stateCounter = 0;
	state();

	//test timer here
	
	QTimer *resUsageTimer = new QTimer(this);
	connect(resUsageTimer, SIGNAL(timeout()), this, SLOT(updateResUsage()));
	resUsageTimer->start(1000);
	
	/*PROCESS_MEMORY_COUNTERS pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
	SIZE_T virtualMemUsedByMe = pmc.PagefileUsage;
	SIZE_T physMemUsedByMe = pmc.WorkingSetSize;
	QString resUsage = tr("%1 MB (%2 MB in pagefile)").arg(QString::number(physMemUsedByMe / 1048576.0)).arg(QString::number(virtualMemUsedByMe / 1048576.0));
	ui.labelResUsage->setText(resUsage);*/
	
	//// Get the list of process identifiers.
	//DWORD aProcesses[1024], cbNeeded, cProcesses;
	//unsigned int i;

	//if (!EnumProcesses(aProcesses, sizeof(aProcesses), &cbNeeded))
	//{
	//	std::cout << "Error getting process handle!\n";
	//}

	//// Calculate how many process identifiers were returned.
	//cProcesses = cbNeeded / sizeof(DWORD);

	//// Print the memory usage for each process
	//for (i = 0; i < cProcesses; i++)
	//{
	//	PrintMemoryInfo(aProcesses[i]);
	//}

}

void QtGuiApplication::updateRenderProgressBar(int newvalue)
{
	newvalue = (newvalue < 0) ? 0 : newvalue;
	newvalue = (newvalue > 100) ? 100 : newvalue;
	std::cout << "[GUI]Render at " << newvalue << " %\n";
	ui.progressRender->setValue(newvalue);
}

void QtGuiApplication::on_radioManual_clicked()
{
	ui.stackedInput->setCurrentIndex(0);
}

void QtGuiApplication::updateRenderProgressDirectly()
{
	float tempvalue1 = 0.0, tempvalue2 = 0.0;
	tracer::getProgress(tempvalue1, tempvalue2);
	int newvalue = tempvalue2*100;
	newvalue = (newvalue < 0) ? 0 : newvalue;
	newvalue = (newvalue > 100) ? 100 : newvalue;
	std::cout << "[GUI]Render at " << newvalue << " %\n";
	ui.progressRender->setValue(newvalue);
}

void QtGuiApplication::timerTest()
{
	std::cout << "Test timer elapsed!\n";
}

void QtGuiApplication::updateResUsage()
{
	PROCESS_MEMORY_COUNTERS pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
	SIZE_T virtualMemUsedByMe = pmc.PagefileUsage;
	SIZE_T physMemUsedByMe = pmc.WorkingSetSize;

	long freevram = 0, totalvram = 0;
	tracer::getVRAMUsageInfo(totalvram, freevram);
	long usedvram = totalvram - freevram;

	QString resUsage = tr("RAM: %1 MB (%2 MB in pagefile), VRAM: %3 MB (%4 MB total)")
		.arg(QString::number(physMemUsedByMe / 1048576.0))
		.arg(QString::number(virtualMemUsedByMe / 1048576.0))
		.arg(QString::number(usedvram / 1048576.0))
		.arg(QString::number(totalvram / 1048576.0));

	ui.labelResUsage->setText(resUsage);
}

void QtGuiApplication::on_radioPicture_clicked()
{
	ui.stackedInput->setCurrentIndex(1);
}

void QtGuiApplication::on_radioList_clicked()
{
	ui.stackedInput->setCurrentIndex(2);
}

void QtGuiApplication::on_pushAddPoint_clicked()
{
	
	AddPointDialog dialog;
	if (dialog.exec() == QDialog::Accepted)
	{
		tracer::PI_LuminousPoint point;
		point.x = dialog.lineX->text().toFloat();
		point.y = dialog.lineY->text().toFloat();
		point.z = dialog.lineZ->text().toFloat();
		point.intensity = dialog.lineIntensity->text().toFloat();
		float tempwavelength = dialog.lineWavelength->text().toFloat();
		tempwavelength = ((float)round(10 * tempwavelength))/10.0;
		point.wavelength = tempwavelength;

		//get a new unique ID
		int uniqueID = rand();
		while (std::find_if(inputTableData.begin(), inputTableData.end(), [&uniqueID](inputTableItem item) {return uniqueID == item.uniqueID; }) != inputTableData.end())
		{
			uniqueID = rand();
		}
		inputTableData.push_back({ uniqueID, point });

		//display to screen
		ui.tableInput->setRowCount(ui.tableInput->rowCount() + 1);
		int currentRow = ui.tableInput->rowCount()-1;
		ui.tableInput->setItem(currentRow, 0, new QTableWidgetItem(tr("(%1, %2, %3)").arg(point.x).arg(point.y).arg(point.z)));
		ui.tableInput->setItem(currentRow, 1, new QTableWidgetItem(tr("%1").arg(point.wavelength)));
		ui.tableInput->setItem(currentRow, 2, new QTableWidgetItem(tr("%1").arg(point.intensity)));
		ui.tableInput->setItem(currentRow, 3, new QTableWidgetItem(tr("%1").arg(uniqueID)));

		//increase point count for the wavelength
		listConfigCompanion::addPoint(point.wavelength);
	}

	stateReset();
}

void QtGuiApplication::on_pushRemovePoint_clicked()
{
	int currentRow = ui.tableInput->currentRow();

	if (currentRow < 0)
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Unknown Error");
		msgBox.setText("No rows to remove!");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
		return;
	}

	int uniqueID = ui.tableInput->item(currentRow, 3)->text().toInt();
	int wavelength = 0;

	auto token = std::find_if(inputTableData.begin(), inputTableData.end(), [&uniqueID](inputTableItem item) {return uniqueID == item.uniqueID; });
	if (token != inputTableData.end())
	{
		wavelength = token->point.wavelength;
		inputTableData.erase(token);
	}
	else
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Unknown Error");
		msgBox.setText("Cannot find the uniqueID of deleted point!");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
	}

	//decrease point count for the wavelength
	listConfigCompanion::removePoint(wavelength);

	ui.tableInput->removeRow(currentRow);

	stateReset();
}

void QtGuiApplication::on_pushClearInput_clicked()
{
	int activePage = ui.stackedInput->currentIndex();
	switch (activePage)
	{
	case 0:
		ui.tableInput->clearContents();
		ui.tableInput->setRowCount(0);
		inputTableData.clear();
		listConfigCompanion::removeAllWavelengths();
		break;
	case 1:
		on_pushClearImage_clicked();
		break;
	case 2:
		//TODO: implement this
		break;
	default:
		break;
	}

	stateReset();
}

void QtGuiApplication::on_pushAddSurface_clicked()
{
	AddSurfaceDialog dialog;
	dialog.pushSelectApoPath->setEnabled(false);
	dialog.lineApoPath->setEnabled(false);
	if (dialog.exec() == QDialog::Accepted)
	{
		float X = dialog.lineX->text().toFloat();
		float Y = dialog.lineY->text().toFloat();
		float Z = dialog.lineZ->text().toFloat();
		float diam = dialog.lineDiam->text().toFloat();
		float R = dialog.lineRadius->text().toFloat();
		float refracI = dialog.lineRefracI->text().toFloat();
		float asph = dialog.lineAsph->text().toFloat();
		int apo = dialog.comboBox->currentIndex();
		QString apoPath;
		const char *c_strApoPath = "";
		QByteArray fileName_ba;
		if (apo == 2) //if custom apodization
		{
			apoPath = dialog.lineApoPath->text(); 
			fileName_ba = apoPath.toLatin1();
			c_strApoPath = fileName_ba.data();
			
			//sample code to convert c string back to qstring and std string
			{
				/*QByteArray testba;
				testba.append(c_strApoPath);*/

				QByteArray testba(c_strApoPath);
				QString outqstr = QString::fromLatin1(testba);
				std::cout << outqstr.toStdString() << "\n";
			}
		}
		
		tableConfigCompanion::addSurface(X, Y, Z, diam, R, refracI, asph, apo, c_strApoPath);
	}
}

void QtGuiApplication::on_pushModifySurface_clicked()
{
	tableConfigCompanion::modifySurfaceAtCurrentRow();
}

void QtGuiApplication::on_pushRemoveSurface_clicked()
{
	tableConfigCompanion::deleteSurfaceAtCurrentRow();
}

void QtGuiApplication::on_pushAcceptConfig_clicked()
{
	//TODO: add some data checking here
	
	tableConfigCompanion::acceptCurrentConfig();

	stateReset();
}

void QtGuiApplication::on_pushClearConfig_clicked()
{
	tableConfigCompanion::clearCurrentConfig();
	stateReset();
}

void QtGuiApplication::on_pushVisualizeConfig_clicked()
{
	tableConfigCompanion::drawCurrentConfig();
}

void QtGuiApplication::on_pushCloneConfig_clicked()
{
	tableConfigCompanion::cloneToCurrentConfig();
}

void QtGuiApplication::on_pushSaveConfig_clicked()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save Optical Configuration"),
		"",
		tr("Optical Configuration Binaries (*.ocb)"));

	if (fileName.isEmpty())
	{
		return;
	}

	tableConfigCompanion::saveCurrentConfig(fileName);
}

void QtGuiApplication::on_pushLoadConfig_clicked()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load Optical Configuration"),
		"",
		tr("Optical Configuration Binaries (*.ocb)"));

	if (fileName.isEmpty())
		return;

	tableConfigCompanion::loadToCurrentConfig(fileName);
}

void QtGuiApplication::on_listConfig_currentItemChanged()
{
	float selectedwavelength = 0;
	if (ui.listConfig->currentItem() != nullptr)
	{
		selectedwavelength = ui.listConfig->currentItem()->text().toFloat();
	}
	else
	{
		return;
	}
	tableConfigCompanion::checkOutWavelength(selectedwavelength);

	if (ui.listConfig->count() > 0) {
		ui.listConfig->currentItem()->setSelected(true);
	}
	ui.listConfig->setFocus();
}

void QtGuiApplication::on_pushCheckData_clicked()
{
	tracer::clearStorage();

	stateReset();

	//start by inputting all points

	//input points from the table
	{
		tracer::PI_Message message;
		int count = 0, count_failed = 0;
		for (auto eachitem : inputTableData)
		{
			message = tracer::addPoint(eachitem.point);
			if (message.code != PI_OK)
			{
				std::cout << "Could not add point (" << eachitem.point.x << ", " << eachitem.point.y << ", " << eachitem.point.z << "), intensity = " << eachitem.point.intensity << ", at " << eachitem.point.wavelength << " nm\n";
				count_failed++;
			}
			else
			{
				count++;
			}
		}

		std::cout << "Adding Point Summary: Tried to add " << inputTableData.size() << " point(s)\n";
		std::cout << count << " point(s) added successfully\n" << count_failed << " point(s) added unsucessfully\n";
	}

	//TODO: input points from the picture
	{
		QString fileName = ui.lineImagePath->text();
		if (fileName.isEmpty() == false)
		{
			QByteArray fileName_ba = fileName.toLatin1();
			const char *c_strFileName = fileName_ba.data();

			float brightness = ui.lineImageBrightness->text().toFloat();
			float posX = ui.lineImagePosX->text().toFloat();
			float posY = ui.lineImagePosY->text().toFloat();
			float posZ = ui.lineImagePosZ->text().toFloat();
			float rotX = ui.lineImageRotX->text().toFloat();
			float rotY = ui.lineImageRotY->text().toFloat();
			float rotZ = ui.lineImageRotZ->text().toFloat();
			float horzSize = ui.lineImageHorzSize->text().toFloat();
			float vertSize = ui.lineImageVertSize->text().toFloat();

			//data check has been done when add image path, no need to add again

			//calling API
			tracer::PI_Message message = tracer::importImage(c_strFileName, posX, posY, posZ, horzSize, vertSize, rotX, rotY, rotZ, brightness);
			if (message.code != PI_OK)
			{
				std::cout << "[GUI] Could not import image!\n";
				QMessageBox msgBox;
				msgBox.setWindowTitle("Input Error");
				msgBox.setText("Could not import image!\n");
				msgBox.setInformativeText(message.detail);
				msgBox.setStandardButtons(QMessageBox::Ok);
				msgBox.setDefaultButton(QMessageBox::Ok);
				msgBox.exec();
			}
		}
		else
		{
			std::cout << "[GUI] No image was added\n";
		}
	}

	//TODO: input points from external file

	//check the image surface
	float imgDiam = ui.lineImageDiam->text().toFloat();
	float imgRadius = ui.lineImageRadius->text().toFloat();
	float imgAngularResol = ui.lineAngularResol->text().toFloat();
	float imgAngularExtend = ui.lineAngularExtend->text().toFloat();
	
	{
		QString errorstr;

		if (imgDiam <= 0)
		{
			errorstr.append("Image Diameter must be positive.\n");
		}

		if (imgRadius >= 0)
		{
			errorstr.append("Image Curvature Radius should be negative\n");
		}

		//TODO: sophisticated check needed here!
		if (imgAngularResol <= 0)
		{
			errorstr.append("Angular Resolution out of range\n");
		}

		if (imgAngularExtend <= 0 || imgAngularExtend > 90.0)
		{
			errorstr.append("Angular Extend out of range between 0 and 90 degree\n");
		}

		if (errorstr.isEmpty() == false)
		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("Input Error");
			msgBox.setText("Please check the following error(s):");
			msgBox.setInformativeText(errorstr);
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
			return;
		}
	}

	float* traceableWavelengths = nullptr;
	int traceableCount = 0;
	float* untraceableWavelengths = nullptr;
	int untraceableCount = 0;
	listConfigCompanion::getTraceableWavelengths(traceableWavelengths, traceableCount, untraceableWavelengths, untraceableCount);

	if (untraceableCount != 0)
	{
		QString errorstr;
		for (int j = 0; j < untraceableCount; j++)
		{
			errorstr.append(tr("Wavelength %1, configuration not valid!").arg(untraceableWavelengths[j]));
		}
		QMessageBox msgBox;
		msgBox.setWindowTitle("Input Error");
		msgBox.setText("Please check the following error(s):");
		msgBox.setInformativeText(errorstr);
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();

		if (traceableCount != 0 && traceableWavelengths != nullptr)
		{
			delete[] traceableWavelengths;
			traceableWavelengths = nullptr;
		}
		if (untraceableCount != 0 && untraceableWavelengths != nullptr)
		{
			delete[] untraceableWavelengths;
			untraceableWavelengths = nullptr;
		}

		return;
	}

	if (traceableCount == 0)
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Error");
		msgBox.setText("No traceable wavelengths!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();

		if (traceableCount != 0 && traceableWavelengths != nullptr)
		{
			delete[] traceableWavelengths;
			traceableWavelengths = nullptr;
		}
		if (untraceableCount != 0 && untraceableWavelengths != nullptr)
		{
			delete[] untraceableWavelengths;
			untraceableWavelengths = nullptr;
		}

		return;
	}

	//clear the output list before adding new
	ui.listOutputWavelength->clear();

	for (int i = 0; i < traceableCount; i++)
	{
		std::list<tracer::PI_Surface> configToAdd;
		if (!tableConfigCompanion::getConfigAt(traceableWavelengths[i], configToAdd))
		{
			continue;
		}
		tracer::PI_Surface imagesurf;
		imagesurf.diameter = imgDiam;
		imagesurf.radius = imgRadius;
		configToAdd.push_back(imagesurf);

		tracer::PI_Surface* argconfig = new tracer::PI_Surface[configToAdd.size()];
		{
			int j = 0;
			for (auto surface : configToAdd)
			{
				argconfig[j] = surface;
				j++;
			}
		}

		tracer::addOpticalConfigAt(traceableWavelengths[i], configToAdd.size(), argconfig, imgAngularResol, imgAngularExtend);

		//add wavelength to the output list
		ui.listOutputWavelength->addItem(new QListWidgetItem(tr("%1").arg(traceableWavelengths[i])));
	}

	if (tracer::checkData().code != PI_OK)
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Core API error");
		msgBox.setText("Check data API returned error!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();

		if (traceableCount != 0 && traceableWavelengths != nullptr)
		{
			delete[] traceableWavelengths;
			traceableWavelengths = nullptr;
		}
		if (untraceableCount != 0 && untraceableWavelengths != nullptr)
		{
			delete[] untraceableWavelengths;
			untraceableWavelengths = nullptr;
		}
		return;
	}

	//TODO: sophisticated check for already added wavelengths
	//...

	QMessageBox msgBox;
	msgBox.setWindowTitle("Info");
	msgBox.setText("Configurations Added OK!\n");
	msgBox.setStandardButtons(QMessageBox::Ok);
	msgBox.setDefaultButton(QMessageBox::Ok);
	msgBox.exec();

	if (traceableCount != 0 && traceableWavelengths != nullptr)
	{
		delete[] traceableWavelengths;
		traceableWavelengths = nullptr;
	}
	if (untraceableCount != 0 && untraceableWavelengths != nullptr)
	{
		delete[] untraceableWavelengths;
		untraceableWavelengths = nullptr;
	}

	stateNext();
}

void QtGuiApplication::on_pushTrace_clicked()
{
	if (tracer::trace().code != PI_OK)
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Core API error");
		msgBox.setText("Trace API returned error!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();

		stateReset();
		return;
	}
	stateNext();
}

void QtGuiApplication::on_pushRender_clicked()
{
	/*
	RenderProgressEmittor emittor;
	connect(&emittor, &RenderProgressEmittor::progressChanges, this, &QtGuiApplication::updateRenderProgressBar);
	auto renderProgressUpdater = [](RenderProgressEmittor* handle) {
		while (true)
		{
			handle->setProgress((int)(100.0*tracer::PI_renderProgress));
			using namespace std::literals::chrono_literals;
			std::this_thread::sleep_for(0.5s);
		}
		};

	std::thread threadRenderProgress(renderProgressUpdater, &emittor);
	*/
	/*
	QThread* thread = new QThread(this);
	RenderProgressEmittor* worker = new RenderProgressEmittor; 
	worker->moveToThread(thread);

	connect(thread, SIGNAL(finished()), worker, SLOT(deleteLater()));
	connect(thread, SIGNAL(started()), worker, SLOT(watchProgress()));
	connect(worker, SIGNAL(progressChanges(int)), this, SLOT(updateRenderProgressBar(int)));

	thread->start();
	*/
	
	std::future<tracer::PI_Message> asyncresult = std::async(std::launch::async, &tracer::render);

	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(timerTest()));
	connect(timer, SIGNAL(timeout()), this, SLOT(updateRenderProgressDirectly()));
	timer->start(200);

	//auto rendermsg = tracer::render();
	//std::thread rendererThread(tracer::render);
	//tracer::render();
	if (false)
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Rendering...");
		msgBox.setText("Please be patient!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
	}
	{
		//std::cout << "[GUI] Renderer is running!\n";
	}
	
	auto rendermsg = asyncresult.get();
	/*
	if (rendermsg.code != PI_OK)
	{
		
		//if (threadRenderProgress.joinable())
		//	threadRenderProgress.join();
		
		//thread->quit();
		//thread->wait();
		
		ui.progressRender->setValue(0);
		timer->stop();

		QMessageBox msgBox;
		msgBox.setWindowTitle("Core API error");
		msgBox.setText("Render API returned error!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();

		stateReset();
		return;
	}
	*/

	//if (rendererThread.joinable()) rendererThread.join();

	ui.progressRender->setValue(100);
	timer->stop();

	
	/*
	thread->quit();
	thread->wait();
	*/
	/*
	if (threadRenderProgress.joinable())
		threadRenderProgress.join();

	emittor.setProgress((int)(75));
	*/
	
	
	//create output image: these codes temporarily lies here
	float* traceableWavelengths = nullptr;
	int traceableCount = 0;
	float* untraceableWavelengths = nullptr;
	int untraceableCount = 0;
	listConfigCompanion::getTraceableWavelengths(traceableWavelengths, traceableCount, untraceableWavelengths, untraceableCount);
	
	for (auto oldID : outputImageIDs)
	{
		tracer::deleteOutputImage(oldID);
	}
	outputImageIDs.clear();

	int newID = 0;
	if (tracer::createOutputImage(traceableCount, traceableWavelengths, newID).code == PI_OK)
	{
		outputImageIDs.push_back(newID);
	}
	else
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Core API error");
		msgBox.setText("Error at creating output image!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
	}
	
	stateNext();
}

void QtGuiApplication::on_pushShowWavelength_clicked()
{
	if (ui.listOutputWavelength->currentItem() != nullptr)
	{
		float wavelength = ui.listOutputWavelength->currentItem()->text().toFloat();
		int count = 1;
		tracer::showRaw(&wavelength, count);
	}
	else
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Info");
		msgBox.setText("Output list is empty!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
	}
}

void QtGuiApplication::on_pushDisplayRGB_clicked()
{
	if (ui.listOutputWavelength->count() != 0)
	{
		//temporary implementation
		if (outputImageIDs.size() != 0)
		{
			tracer::showRGB(outputImageIDs[0]);
		}
		else
		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("Error");
			msgBox.setText("There is no RGB output image!\n");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
		}
	}
	else
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Info");
		msgBox.setText("Output list is empty!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
	}
}

void QtGuiApplication::on_pushSaveRaw_clicked()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save Raw Image"),
		"",
		tr("Textfile (*.txt)"));

	if (fileName.isEmpty())
	{
		return;
	}

	QByteArray fileName_ba = fileName.toLatin1();
	const char *c_strFileName = fileName_ba.data();
	
	//temporary implementation
	if (outputImageIDs.size() != 0)
	{
		tracer::saveRaw(c_strFileName, outputImageIDs[0]);
	}
	else
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Error");
		msgBox.setText("There is no RGB output image!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
	}
}

void QtGuiApplication::on_pushSaveRGB_clicked()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save RGB Image"),
		"",
		tr("Image Files (*.jpg *.png )"));

	if (fileName.isEmpty())
	{
		return;
	}

	QByteArray fileName_ba = fileName.toLatin1();
	const char *c_strFileName = fileName_ba.data();
	
	//temporary implementation
	if (outputImageIDs.size() != 0)
	{
		tracer::saveRGB(c_strFileName, outputImageIDs[0]);
	}
	else
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Error");
		msgBox.setText("There is no RGB output image!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
	}
}

void QtGuiApplication::on_pushVisualizeRGB_clicked()
{
	if (ui.checkShowOpticalSurfaces->checkState() == Qt::Checked)
	{
		if (ui.listOutputWavelength->currentItem() != nullptr)
		{
			float wavelength = ui.listOutputWavelength->currentItem()->text().toFloat();
			int count = 1;
			tracer::drawOpticalConfig(wavelength);
		}
		else
		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("Info");
			msgBox.setText("Output list is empty!\n");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
		}
	}
	else
	{
		tracer::drawImage(outputImageIDs[0]);
	}
}

void QtGuiApplication::on_actionTest_triggered()
{
	tracer::test();
}

void QtGuiApplication::on_actionConsoleOut_triggered()
{
	if (w2 != nullptr)
		delete w2;
	w2 = new ConsoleOut();
	w2->show();
}

void QtGuiApplication::on_actionPreferences_triggered()
{
	PreferenceDialog dialog;
	dialog.exec();
}

void QtGuiApplication::on_pushSelectImage_clicked()
{
	//first check the image parameters

	float brightness = ui.lineImageBrightness->text().toFloat();
	float posZ = ui.lineImagePosZ->text().toFloat();
	float horzSize = ui.lineImageHorzSize->text().toFloat();
	float vertSize = ui.lineImageVertSize->text().toFloat();
	//float wavelengthR = ui.lineImageRedWavelength->text().toFloat();
	//float wavelengthG = ui.lineImageGreenWavelength->text().toFloat();
	//float wavelengthB = ui.lineImageBlueWavelength->text().toFloat();

	////rectify the wavelength number
	//float tempwavelengthR = ((float)round(10 * wavelengthR)) / 10.0;
	//ui.lineImageRedWavelength->setText(QString::number(tempwavelengthR));
	//float tempwavelengthG = ((float)round(10 * wavelengthG)) / 10.0;
	//ui.lineImageGreenWavelength->setText(QString::number(tempwavelengthG));
	//float tempwavelengthB = ((float)round(10 * wavelengthB)) / 10.0;
	//ui.lineImageBlueWavelength->setText(QString::number(tempwavelengthB));
	
	{
		QString errorstr;

		if (brightness <= 0.0 || brightness > 1.0)
		{
			errorstr.append("Image must have  brightness value between 0 and 1.0!\n");
		}

		if (posZ <= 0)
		{
			errorstr.append("Z position should be positive!\n");
		}

		if (horzSize <= 0)
		{
			errorstr.append("Horizontal size of image should be positive\n");
		}

		if (vertSize <= 0)
		{
			errorstr.append("Vertical size of image should be positive\n");
		}

		/*if (wavelengthR < 380.0 || wavelengthR > 830.0)
		{
			errorstr.append("Red wavelength outside visible range.\n");
		}

		if (wavelengthG < 380.0 || wavelengthG > 830.0)
		{
			errorstr.append("Green wavelength outside visible range.\n");
		}

		if (wavelengthB < 380.0 || wavelengthB > 830.0)
		{
			errorstr.append("Blue wavelength outside visible range.\n");
		}*/

		if (errorstr.isEmpty() == false)
		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("Input Error");
			msgBox.setText("Please check the following error(s) before continue:");
			msgBox.setInformativeText(errorstr);
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
			return;
		}
	}

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Image"), "", tr("Image Files (*.png *.jpg *.bmp)"));

	if (fileName.isEmpty())
		return;

	ui.lineImagePath->setText(fileName);

	//make all edits read only, because i don't want to re-perform the data check at the on_pushDataCheckin_clicked()
	ui.lineImageBrightness->setReadOnly(true);
	ui.lineImagePosX->setReadOnly(true);
	ui.lineImagePosY->setReadOnly(true);
	ui.lineImagePosZ->setReadOnly(true);
	ui.lineImageRotX->setReadOnly(true);
	ui.lineImageRotY->setReadOnly(true);
	ui.lineImageRotZ->setReadOnly(true);
	ui.lineImageHorzSize->setReadOnly(true);
	ui.lineImageVertSize->setReadOnly(true);

	//get the primaries and add them to the wavelength list
	float wavelengths[3];
	tracer::getImagePrimaryWavelengths(wavelengths[0], wavelengths[1], wavelengths[2]);
	listConfigCompanion::addWavelength(wavelengths[0]);
	listConfigCompanion::addWavelength(wavelengths[1]);
	listConfigCompanion::addWavelength(wavelengths[2]);
}

void QtGuiApplication::on_pushClearImage_clicked()
{
	ui.lineImagePath->clear();

	float wavelengths[3];
	tracer::getImagePrimaryWavelengths(wavelengths[0], wavelengths[1], wavelengths[2]);
	listConfigCompanion::removeWavelength(wavelengths[0]);
	listConfigCompanion::removeWavelength(wavelengths[1]);
	listConfigCompanion::removeWavelength(wavelengths[2]);

	ui.lineImageBrightness->setReadOnly(false);
	ui.lineImagePosX->setReadOnly(false);
	ui.lineImagePosY->setReadOnly(false);
	ui.lineImagePosZ->setReadOnly(false);
	ui.lineImageRotX->setReadOnly(false);
	ui.lineImageRotY->setReadOnly(false);
	ui.lineImageRotZ->setReadOnly(false);
	ui.lineImageHorzSize->setReadOnly(false);
	ui.lineImageVertSize->setReadOnly(false);
}

void tableConfigCompanion::checkOutWavelength(float wavelength)
{
	auto token = std::find_if(configs.begin(), configs.end(), [wavelength](wavelengthAndConfig& holder) {return holder.wavelength == wavelength; });
	if (token == configs.end())
	{
		if (listConfigCompanion::hasWavelength(wavelength))
		{
			//add new config
			wavelengthAndConfig newitem;
			newitem.wavelength = wavelength;
			configs.push_back(newitem);
			/*
			QMessageBox msgBox;
			msgBox.setWindowTitle("Info");
			msgBox.setText(QObject::tr("New config created at %1 nm!").arg(wavelength));
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
			*/
			token = std::find_if(configs.begin(), configs.end(), [wavelength](wavelengthAndConfig& holder) {return holder.wavelength == wavelength; });

		}
		else
		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("Error");
			msgBox.setText("Wavelength does not exist!\nPlease add at least an input point!");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
		}
	}
	currentWavelength = token->wavelength;
	currentConfig = (token->surfaces);

	p_table->clearContents();
	p_table->setRowCount(currentConfig.size());

	if (currentConfig.empty() == false)
	{
		int i = 0;
		for (auto token2 = currentConfig.begin(); token2!=currentConfig.end(); token2++)
		{
			tracer::PI_Surface& surface = (*token2);
			p_table->setItem(i, 0, new QTableWidgetItem(QObject::tr("(%1, %2, %3)").arg(surface.x).arg(surface.y).arg(surface.z)));
			p_table->setItem(i, 1, new QTableWidgetItem(QObject::tr("%1").arg(surface.diameter)));
			p_table->setItem(i, 2, new QTableWidgetItem(QObject::tr("%1").arg(surface.radius)));
			p_table->setItem(i, 3, new QTableWidgetItem(QObject::tr("%1").arg(surface.refractiveIndex)));
			p_table->setItem(i, 4, new QTableWidgetItem(QObject::tr("%1").arg(surface.asphericity)));
			QString apoName;
			switch (surface.apodization)
			{
			case 2:
				apoName = "Custom";
				break;
			case 1:
				apoName = "Bartlett";
				break;
			case 0:
			default:
				apoName = "Uniform";
				break;
			}
			p_table->setItem(i, 5, new QTableWidgetItem(apoName));

			i++;
		}
	}
}

bool tableConfigCompanion::addSurface(float X, float Y, float Z, float diam, float R, float refracI, float asph, int apo, const char* apoPath)
{
	tracer::PI_Surface newsurface;
	newsurface.x = X;
	newsurface.y = Y;
	newsurface.z = Z;
	newsurface.diameter = diam;
	newsurface.radius = R;
	newsurface.refractiveIndex = refracI;
	newsurface.asphericity = asph;
	newsurface.customApoPath = "";

	//apodization translator
	QString apoName;
	QByteArray baToAdd;
	switch (apo)
	{
	case 2:
		apoName = "Custom";
		newsurface.apodization = PI_APD_CUSTOM;
		baToAdd.append(apoPath);
		apoPathList.push_back(baToAdd);
		{
			auto newtoken = apoPathList.end();
			newtoken--;
			newsurface.customApoPath = newtoken->data();
		}
		/*{
			QByteArray testba(newsurface.customApoPath);
			QString outqstr = QString::fromLatin1(testba);
			std::cout << outqstr.toStdString() << "\n";
		}*/
		break;
	case 1:
		apoName = "Bartlett";
		newsurface.apodization = PI_APD_BARTLETT;
		break;
	case 0:
	default:
		apoName = "Uniform";
		newsurface.apodization = PI_APD_UNIFORM;
		break;
	}
	

	if (currentConfig.size() != 0)
	{
		//first make sure there is no duplication
		for (auto eachsurface : currentConfig)
		{
			if (eachsurface.x == X &&
				eachsurface.y == Y &&
				eachsurface.z == Z &&
				eachsurface.radius == R &&
				eachsurface.asphericity == asph)
			{
				QMessageBox msgBox;
				msgBox.setWindowTitle("Input error");
				msgBox.setText("Surface cannot have the same geometry and position as existing surface!\n");
				msgBox.setStandardButtons(QMessageBox::Ok);
				msgBox.setDefaultButton(QMessageBox::Ok);
				msgBox.exec();

				//save a bit space by throw away unused string
				apoPathList.pop_back();

				return false;
			}
		}

		int index = 0;
		auto token = currentConfig.begin();
		for (; token != currentConfig.end(); token++)
		{
			if (token->z < newsurface.z)
			{
				currentConfig.insert(token, newsurface);
				break;
			}
			index++;
		}

		if (token == currentConfig.end())
		{
			currentConfig.insert(currentConfig.end(), newsurface);
		}

		//p_table->setRowCount(currentConfig.size() + 1);
		p_table->insertRow(index);

		p_table->setItem(index, 0, new QTableWidgetItem(QObject::tr("(%1, %2, %3)").arg(newsurface.x).arg(newsurface.y).arg(newsurface.z)));
		p_table->setItem(index, 1, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.diameter)));
		p_table->setItem(index, 2, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.radius)));
		p_table->setItem(index, 3, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.refractiveIndex)));
		p_table->setItem(index, 4, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.asphericity)));
		p_table->setItem(index, 5, new QTableWidgetItem(apoName));

	}
	else
	{
		currentConfig.insert(currentConfig.begin(), newsurface);
		p_table->setRowCount(1);
		p_table->setItem(0, 0, new QTableWidgetItem(QObject::tr("(%1, %2, %3)").arg(newsurface.x).arg(newsurface.y).arg(newsurface.z)));
		p_table->setItem(0, 1, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.diameter)));
		p_table->setItem(0, 2, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.radius)));
		p_table->setItem(0, 3, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.refractiveIndex)));
		p_table->setItem(0, 4, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.asphericity)));
		p_table->setItem(0, 5, new QTableWidgetItem(apoName));
	}
	
	return true;
}

bool tableConfigCompanion::modifySurfaceAtCurrentRow()
{
	//preliminary check
	int rowCount = p_table->rowCount();
	if (rowCount == 0)
	{
		return false;
	}

	//get the surface at current row
	int currentRow = 0;
	currentRow = p_table->currentRow();

	auto token = currentConfig.begin();
	for (int i = 0; i < currentRow; i++)
	{
		token++;
		if (token == currentConfig.end())
		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("Error");
			msgBox.setText("Count mismatch between tableConfig and its companion!\n");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
			return false;
		}
	}

	//create dialog
	AddSurfaceDialog dialog;
	dialog.pushSelectApoPath->setEnabled(false);
	dialog.lineApoPath->setEnabled(false);

	// and put its data to the dialog
	dialog.lineX->setText(QString::number(token->x));
	dialog.lineY->setText(QString::number(token->y));
	dialog.lineZ->setText(QString::number(token->z));
	dialog.lineDiam->setText(QString::number(token->diameter));
	dialog.lineRadius->setText(QString::number(token->radius));
	dialog.lineRefracI->setText(QString::number(token->refractiveIndex));
	dialog.lineAsph->setText(QString::number(token->asphericity));
	dialog.comboBox->setCurrentIndex(token->apodization);
	if (token->apodization == PI_APD_CUSTOM)
	{
		dialog.pushSelectApoPath->setEnabled(true);
		dialog.lineApoPath->setEnabled(true);
		
		QByteArray testba;
		testba.append(token->customApoPath);
		QString outqstr = QString::fromLatin1(testba);
		dialog.lineApoPath->setText(outqstr);
	}
	
	if (dialog.exec() == QDialog::Accepted)
	{
		float X = dialog.lineX->text().toFloat();
		float Y = dialog.lineY->text().toFloat();
		float Z = dialog.lineZ->text().toFloat();
		float diam = dialog.lineDiam->text().toFloat();
		float R = dialog.lineRadius->text().toFloat();
		float refracI = dialog.lineRefracI->text().toFloat();
		float asph = dialog.lineAsph->text().toFloat();
		int apo = dialog.comboBox->currentIndex();
		const char* apoPath = "";
		QByteArray fileName_ba;
		if (apo == 2) //if custom apodization
		{
			QString qtApoPath = dialog.lineApoPath->text();
			fileName_ba = qtApoPath.toLatin1();
			apoPath = fileName_ba.data();
		}

		//if the modified value stays the same, return
		if (token->x == X &&
			token->y == Y &&
			token->z == Z &&
			token->diameter == diam &&
			token->radius == R &&
			token->refractiveIndex == refracI &&
			token->asphericity == asph &&
			token->apodization == apo &&
			strcmp(token->customApoPath, apoPath) == 0
			)
		{
			return true;
		}

		//scan for duplication, except at the current surface
		for (auto token2 = currentConfig.begin(); token2 != currentConfig.end(); token2++)
		{
			if (token2 == token)
			{
				continue;
			}
			auto eachsurface = *token2;
			if (eachsurface.x == X &&
				eachsurface.y == Y &&
				eachsurface.z == Z &&
				eachsurface.radius == R &&
				eachsurface.asphericity == asph)
			{
				QMessageBox msgBox;
				msgBox.setWindowTitle("Input error");
				msgBox.setText("Surface cannot have the same geometry and position as existing surface!\n");
				msgBox.setStandardButtons(QMessageBox::Ok);
				msgBox.setDefaultButton(QMessageBox::Ok);
				msgBox.exec();
				return false;
			}
		}
		
		//if there is no duplication, then the ADD call should be running OK, and it's safe to delete the current surface
		tableConfigCompanion::deleteSurfaceAtCurrentRow();

		//call add
		tableConfigCompanion::addSurface(X, Y, Z, diam, R, refracI, asph, apo, apoPath);

		return true;
	}
	else
	{
		return false;
	}
}

bool tableConfigCompanion::deleteSurfaceAtCurrentRow()
{
	int rowCount = p_table->rowCount();
	if (rowCount == 0)
	{
		//nothing to delete, return
		return false;
	}

	int currentRow = 0;
	currentRow = p_table->currentRow();

	auto token = currentConfig.begin();
	for (int i = 0; i < currentRow; i++)
	{
		token++;
		if (token == currentConfig.end()) 
		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("Error");
			msgBox.setText("Count mismatch between tableConfig and its companion!\n");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
			return false;
		}
	}

	currentConfig.erase(token);
	p_table->removeRow(currentRow);
	p_table->setRowCount(rowCount - 1);

	return true;
}

bool tableConfigCompanion::acceptCurrentConfig()
{
	if (currentConfig.size() != 0)
	{
		float wavelength = currentWavelength;
		auto token = std::find_if(configs.begin(), configs.end(), [wavelength](wavelengthAndConfig& holder) {return holder.wavelength == wavelength; });

		if (token != configs.end())
		{
			token->surfaces = currentConfig;
		}

		listConfigCompanion::setHasConfig(wavelength, true);

		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("Info");
			msgBox.setText("Configuration saved!\n");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
		}

		return true;
	}
	else
	{
		listConfigCompanion::setHasConfig(currentWavelength, false);

		QMessageBox msgBox;
		msgBox.setWindowTitle("Info");
		msgBox.setText("Configuration is empty!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();

		return false;
	}
}

bool tableConfigCompanion::clearCurrentConfig()
{
	float wavelength = currentWavelength;
	auto token = std::find_if(configs.begin(), configs.end(), [wavelength](wavelengthAndConfig& holder) {return holder.wavelength == wavelength; });

	if (token != configs.end())
	{
		token->surfaces.clear();
	}

	currentConfig.clear();

	listConfigCompanion::setHasConfig(wavelength, false);

	p_table->clearContents();
	p_table->setRowCount(0);

	return true;
}

bool tableConfigCompanion::cloneToCurrentConfig()
{
	CloneConfigDialog dialog;

	std::vector<float> wavelengthsToShow;

	for (auto eachconfig : configs)
	{
		wavelengthsToShow.push_back(eachconfig.wavelength);
		dialog.comboCloneWavelength->addItem(QString::number(eachconfig.wavelength).append(" nm"));
	}

	float selectedWavelength;

	if (dialog.exec() == QDialog::Accepted)
	{
		int selectedInt = dialog.comboCloneWavelength->currentIndex();
		if (selectedInt >= 0 && selectedInt <= wavelengthsToShow.size())
		{
			selectedWavelength = wavelengthsToShow[selectedInt];
		}
		else
		{
			return false;
		}
	}

	//clear and clone
	currentConfig.clear();
	auto token = std::find_if(configs.begin(), configs.end(), [selectedWavelength](wavelengthAndConfig& holder) {return holder.wavelength == selectedWavelength; });
	if (token == configs.end())
	{
		return false;
	}
	currentConfig = token->surfaces;

	//populate the table
	p_table->clearContents();
	p_table->setRowCount(currentConfig.size());

	if (currentConfig.empty() == false)
	{
		int i = 0;
		for (auto token2 = currentConfig.begin(); token2 != currentConfig.end(); token2++)
		{
			tracer::PI_Surface& surface = (*token2);
			p_table->setItem(i, 0, new QTableWidgetItem(QObject::tr("(%1, %2, %3)").arg(surface.x).arg(surface.y).arg(surface.z)));
			p_table->setItem(i, 1, new QTableWidgetItem(QObject::tr("%1").arg(surface.diameter)));
			p_table->setItem(i, 2, new QTableWidgetItem(QObject::tr("%1").arg(surface.radius)));
			p_table->setItem(i, 3, new QTableWidgetItem(QObject::tr("%1").arg(surface.refractiveIndex)));
			p_table->setItem(i, 4, new QTableWidgetItem(QObject::tr("%1").arg(surface.asphericity)));
			QString apoName;
			switch (surface.apodization)
			{
			case 2:
				apoName = "Custom";
				break;
			case 1:
				apoName = "Bartlett";
				break;
			case 0:
			default:
				apoName = "Uniform";
				break;
			}
			p_table->setItem(i, 5, new QTableWidgetItem(apoName));

			i++;
		}
	}

	return false;
}

bool tableConfigCompanion::saveCurrentConfig(QString path)
{
	if (currentConfig.size() == 0)
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Info");
		msgBox.setText("Current configuration is empty, there is nothing to save!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
		return false;
	}

	QFile file(path);
	if (!file.open(QIODevice::WriteOnly))
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Error");
		msgBox.setText(QObject::tr("Cannot save to file %1 \n").arg(path));
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
		return false;
	}
	QDataStream out(&file);
	
	out << currentWavelength;
	out << (int)currentConfig.size();
	for (auto eachsurface : currentConfig)
	{
		out << eachsurface.x;
		out << eachsurface.y;
		out << eachsurface.z;
		out << eachsurface.diameter;
		out << eachsurface.radius;
		out << eachsurface.refractiveIndex;
		out << eachsurface.asphericity;
		if (eachsurface.apodization != PI_APD_CUSTOM)
		{
			out << eachsurface.apodization;
		}
		else
		{
			out << PI_APD_UNIFORM;
			// doesn't make sense to save the path to custom apodization file, because the file could get missing the next time the config gets loaded
		}
	}
	file.close();
	return true;
}

bool tableConfigCompanion::loadToCurrentConfig(QString path)
{
	QFile file(path);
	if (!file.open(QIODevice::ReadOnly))
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Error");
		msgBox.setText(QObject::tr("Cannot open file %1 \n").arg(path));
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
		return false;
	}
	QDataStream in(&file);

	float fileWavelength;
	in >> fileWavelength;

	if (fileWavelength != currentWavelength)
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Warning");
		msgBox.setText("The wavelength in file is not the same as the current wavelength!\nDo you want to continue?");
		msgBox.setStandardButtons(QMessageBox::Yes| QMessageBox::No);
		msgBox.setDefaultButton(QMessageBox::No);
		if (msgBox.exec() == QMessageBox::No)
			return false;
	}

	currentConfig.clear();
	p_table->clearContents();
	p_table->setRowCount(0);

	int count = 0;
	in >> count;
	for (int i = 0; i < count; i++)
	{
		tracer::PI_Surface newsurface;
		in >> newsurface.x;
		in >> newsurface.y;
		in >> newsurface.z;
		in >> newsurface.diameter;
		in >> newsurface.radius;
		in >> newsurface.refractiveIndex;
		in >> newsurface.asphericity;
		in >> newsurface.apodization;
		addSurface(newsurface.x, newsurface.y, newsurface.z, newsurface.diameter, newsurface.radius, newsurface.refractiveIndex, newsurface.asphericity, newsurface.apodization);
	}

	return true;
}

bool tableConfigCompanion::clearConfigAt(float wavelength)
{
	auto token = std::find_if(configs.begin(), configs.end(), [wavelength](wavelengthAndConfig& holder) {return holder.wavelength == wavelength; });
	if (token != configs.end())
	{
		configs.erase(token);
	}

	if (wavelength == currentWavelength)
	{
		if (configs.size() == 0)
		{
			currentWavelength = 0;
			currentConfig.clear();
			p_table->clearContents();
			p_table->setRowCount(0);
		}
		else
		{
			auto token = configs.begin();
			currentWavelength = token->wavelength;
			currentConfig = token->surfaces;
			checkOutWavelength(wavelength);
		}
	}
	return true;
}

bool tableConfigCompanion::getConfigAt(float wavelength, std::list<tracer::PI_Surface>& output)
{
	auto token = std::find_if(configs.begin(), configs.end(), [wavelength](wavelengthAndConfig& holder) {return holder.wavelength == wavelength; });

	if (token == configs.end())
	{
		return false;
	}
	else
	{
		output = token->surfaces;
		return true;
	}
}

void tableConfigCompanion::clearApoPathList()
{
	apoPathList.clear();
}

void tableConfigCompanion::clearAllData()
{
	currentConfig.clear();
	configs.clear();
	apoPathList.clear();
}

void tableConfigCompanion::drawCurrentConfig()
{
	//TODO later (non critical): should develop a new set of API for this...
	//... so that draw data comes directly from the UI, not from the storage
	//current workaround: only enable the button once the check_data has been called (draw from storage)
	tracer::drawOpticalConfig(currentWavelength, false, true);
}

void listConfigCompanion::removeWavelength(float wavelength)
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

		tableConfigCompanion::clearConfigAt(wavelength);
	}
}

void listConfigCompanion::removeAllWavelengths()
{
	p_widget->clear();
	for (auto eachwavelength : wavelengths)
	{
		tableConfigCompanion::clearConfigAt(eachwavelength.wavelength);
	}
	wavelengths.clear();

	tableConfigCompanion::clearAllData();
}