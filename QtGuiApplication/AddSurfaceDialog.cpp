#include "AddSurfaceDialog.h"
#include "qmessagebox.h"
#include <qfiledialog.h>


AddSurfaceDialog::AddSurfaceDialog(QWidget *parent)
	: QDialog(parent)
{
	setupUi(this);
}

AddSurfaceDialog::~AddSurfaceDialog()
{
}

void AddSurfaceDialog::on_comboBox_currentIndexChanged()
{
	if (comboBox->currentIndex() == 2)
	{
		pushSelectApoPath->setEnabled(true);
		lineApoPath->setEnabled(true);
	}
	else
	{
		pushSelectApoPath->setEnabled(false);
		lineApoPath->setEnabled(false);
	}
}

void AddSurfaceDialog::on_pushSelectApoPath_clicked()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load Custom Apodization Image"),
		"",
		tr("Image Files (*.jpg *.png *.bmp)"));

	if (fileName.isEmpty())
		return;

	lineApoPath->setText(fileName);
}

void AddSurfaceDialog::on_pushAddSurface_clicked()
{
	bool dataOK = false;
	QString errorstr;

	float Z = this->lineZ->text().toFloat();
	float diam = this->lineDiam->text().toFloat();
	float R = this->lineRadius->text().toFloat();
	float refracI = this->lineRefracI->text().toFloat();
	int apo = this->comboBox->currentIndex();
	QString apoPath = lineApoPath->text();

	if (Z <= 0)
	{
		errorstr.append("Z must be positive (Refractive surface must lie in front of image surface).\n");
	}

	if (diam <= 0)
	{
		errorstr.append("Diameter must be a positive number.\n");
	}

	if (diam > 2.0*abs(R))
	{
		errorstr.append("Diameter must smaller than twice the curvature radius.\n");
	}

	if (refracI < 1.0)
	{
		errorstr.append("Normal optical material should have refractive index not smaller than 1.\n");
	}

	if (apo == 2 && apoPath.isEmpty())
	{
		errorstr.append("Please select an image file for the custom apodization!\n");
	}

	/*
	if (apo != 0)
	{
		errorstr.append("Apodization not yet supported.\n");
	}
	*/

	if (errorstr.isEmpty())
	{
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
		return;
	}

	if (dataOK)
	{
		this->accept();
	}
}
