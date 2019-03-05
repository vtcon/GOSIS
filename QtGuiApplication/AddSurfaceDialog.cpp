#include "AddSurfaceDialog.h"
#include "qmessagebox.h"

AddSurfaceDialog::AddSurfaceDialog(QWidget *parent)
	: QDialog(parent)
{
	setupUi(this);
}

AddSurfaceDialog::~AddSurfaceDialog()
{
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

	if (apo != 0)
	{
		errorstr.append("Apodization not yet supported.\n");
	}

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
