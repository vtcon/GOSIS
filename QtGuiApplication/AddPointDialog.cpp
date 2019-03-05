#include "AddPointDialog.h"
#include "qmessagebox.h"
#include "QtGuiApplication.h"

//extern std::vector<float> inputWavelengths;
extern listConfigCompanion wavelengthList;

AddPointDialog::AddPointDialog(QWidget *parent)
	: QDialog(parent)
{
	setupUi(this);
}

AddPointDialog::~AddPointDialog()
{
}

void AddPointDialog::on_pushButton_clicked()
{
	bool dataOK = false;
	QString errorstr;

	float intensity = this->lineIntensity->text().toFloat();
	float wavelength = this->lineWavelength->text().toFloat();
	float Z = this->lineZ->text().toFloat();

	if (intensity <= 0)
	{
		errorstr.append("Intensity must be a positive number.\n");
	}

	if (Z <= 0)
	{
		errorstr.append("Z coordinate must be a positive number.\n");
	}
	
	bool wavelengthAddAccept = false;

	if (wavelength < 380.0 || wavelength > 830.0)
	{
		errorstr.append("Wavelength outside visible range.\n");
	}
	else
	{
		if (wavelengthList.isExist(wavelength) == false)
		{
			QMessageBox msgBox;
			msgBox.setWindowTitle("New Wavelength");
			msgBox.setText("The entered point has a new wavelength");
			msgBox.setInformativeText("Do you want to add a new wavelength to the list?");
			msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
			msgBox.setDefaultButton(QMessageBox::No);
			int ret = msgBox.exec();
			switch (ret)
			{
			case QMessageBox::Yes:
				wavelengthAddAccept = true;
				break;
			default:
			case QMessageBox::No:
				errorstr.append("Please revise your wavelength!\n");
				break;
			}
		}
	}
	

	if (errorstr.isEmpty())
	{
		float tempwavelength = wavelength;
		tempwavelength = ((float)round(10 * tempwavelength)) / 10.0;
		wavelengthList.addWavelength(tempwavelength);
		dataOK = true;
	}
	else
	{
		QMessageBox msgBox;
		msgBox.setWindowTitle("Input Error");
		msgBox.setText("Please check the following error(s):");
		msgBox.setInformativeText(errorstr);
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		int ret = msgBox.exec();
	}

	if (dataOK)
	{
		this->accept();
	}
}