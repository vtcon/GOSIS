#include "QtGuiApplication.h"

#include "AddPointDialog.h"
#include "AddSurfaceDialog.h"

#include <qobject.h>
#include <qthread.h>
#include "qmessagebox.h"
#include <QFileDialog>
#include <qtimer.h>

#include <QTextStream>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <thread>

//supporting structures
#include "RenderProgressEmittor.h"

//global variables
//std::vector<float> inputWavelengths;
listConfigCompanion wavelengthList;
tableConfigCompanion configTable;
std::vector<int> outputImageIDs;

QtGuiApplication::QtGuiApplication(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	ui.tableInput->setColumnHidden(3, true);

	//
	m_qd = new QDebugStream(std::cout, ui.textEditProcess); //Redirect Console output to QTextEdit
	//m_qd->QDebugStream::registerQDebugMessageHandler(); //Redirect qDebug() output to QTextEdit


	//attach global variables to widgets
	wavelengthList.attachTo(ui.listConfig);
	configTable.attachTo(ui.tableConfig);

	//start the state machine
	stateCounter = 0;
	state();

	//test timer here
	/*
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(timerTest()));
	timer->start(500);
	*/
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
	int newvalue = (int)(100.0*tracer::PI_renderProgress);
	newvalue = (newvalue < 0) ? 0 : newvalue;
	newvalue = (newvalue > 100) ? 100 : newvalue;
	std::cout << "[GUI]Render at " << newvalue << " %\n";
	ui.progressRender->setValue(newvalue);
}

void QtGuiApplication::timerTest()
{
	std::cout << "Time Elapsed!\n";
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

	wavelengthList.removePoint(wavelength);
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
		break;
	case 1:
		//TODO: implement this
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

		configTable.addSurface(X, Y, Z, diam, R, refracI, asph, apo);
	}
}

void QtGuiApplication::on_pushAcceptConfig_clicked()
{
	//TODO: add some data checking here
	
	configTable.acceptCurrentConfig();

	stateReset();
}

void QtGuiApplication::on_pushClearConfig_clicked()
{
	configTable.clearCurrentConfig();
	stateReset();
}

void QtGuiApplication::on_listConfig_currentItemChanged()
{
	float selectedwavelength = ui.listConfig->currentItem()->text().toFloat();
	configTable.checkOutWavelength(selectedwavelength);

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
	wavelengthList.getTraceableWavelengths(traceableWavelengths, traceableCount, untraceableWavelengths, untraceableCount);

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
		if (!configTable.getConfigAt(traceableWavelengths[i], configToAdd))
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
	/*
	QTimer *timer = new QTimer(this);
	//connect(timer, SIGNAL(timeout()), this, SLOT(timerTest()));
	connect(timer, SIGNAL(timeout()), this, SLOT(updateRenderProgressDirectly()));
	timer->start();
	*/
	if (tracer::render().code != PI_OK)
	{
		/*
		if (threadRenderProgress.joinable())
			threadRenderProgress.join();
		*/
		/*
		thread->quit();
		thread->wait();
		*/
		//timer->stop();

		QMessageBox msgBox;
		msgBox.setWindowTitle("Core API error");
		msgBox.setText("Render API returned error!\n");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();

		stateReset();
		return;
	}
	//timer->stop();
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
	wavelengthList.getTraceableWavelengths(traceableWavelengths, traceableCount, untraceableWavelengths, untraceableCount);
	
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
		tr("Image Files (*.png *.jpg)"));

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

void QtGuiApplication::on_actionTest_triggered()
{
	tracer::test();
}

void tableConfigCompanion::checkOutWavelength(float wavelength)
{
	auto token = std::find_if(configs.begin(), configs.end(), [wavelength](wavelengthAndConfig& holder) {return holder.wavelength == wavelength; });
	if (token == configs.end())
	{
		if (wavelengthList.hasWavelength(wavelength))
		{
			//add new config
			wavelengthAndConfig newitem;
			newitem.wavelength = wavelength;
			configs.push_back(newitem);

			QMessageBox msgBox;
			msgBox.setWindowTitle("Info");
			msgBox.setText(QObject::tr("New config created at %1 nm!").arg(wavelength));
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();

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
			//p_table->setItem(i, 5, new QTableWidgetItem(QObject::tr("%1").arg(surface.apodization)));
			p_table->setItem(i, 5, new QTableWidgetItem(QObject::tr("Uniform")));

			i++;
		}
	}
}

bool tableConfigCompanion::addSurface(float X, float Y, float Z, float diam, float R, float refracI, float asph, int apo)
{
	tracer::PI_Surface newsurface;
	newsurface.x = X;
	newsurface.y = Y;
	newsurface.z = Z;
	newsurface.diameter = diam;
	newsurface.radius = R;
	newsurface.refractiveIndex = refracI;
	newsurface.asphericity = asph;
	//newsurface.apodization = apo;

	if (currentConfig.size() != 0)
	{
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

		p_table->setRowCount(currentConfig.size() + 1);
		p_table->insertRow(index);

		p_table->setItem(index, 0, new QTableWidgetItem(QObject::tr("(%1, %2, %3)").arg(newsurface.x).arg(newsurface.y).arg(newsurface.z)));
		p_table->setItem(index, 1, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.diameter)));
		p_table->setItem(index, 2, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.radius)));
		p_table->setItem(index, 3, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.refractiveIndex)));
		p_table->setItem(index, 4, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.asphericity)));
		//p_table->setItem(index, 5, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.apodization)));
		p_table->setItem(index, 5, new QTableWidgetItem(QObject::tr("Uniform")));

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
		//p_table->setItem(0, 5, new QTableWidgetItem(QObject::tr("%1").arg(newsurface.apodization)));
		p_table->setItem(0, 5, new QTableWidgetItem(QObject::tr("Uniform")));
	}
	
	return false;
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

		wavelengthList.setHasConfig(wavelength, true);
		return true;
	}
	else
	{
		wavelengthList.setHasConfig(currentWavelength, false);

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

	wavelengthList.setHasConfig(wavelength, false);

	p_table->clearContents();
	p_table->setRowCount(0);

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

