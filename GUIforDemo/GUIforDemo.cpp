#include "GUIforDemo.h"
#include "AddPointDialog.h"
#include "AddSurfaceDialog.h"
#include <QFileDialog>

#include "ProgramInterface.h"

GUIforDemo::GUIforDemo(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

void GUIforDemo::on_radioButton_clicked()
{
	ui.stackedWidget->setCurrentIndex(0);
}

void GUIforDemo::on_radioButton_2_clicked()
{
	ui.stackedWidget->setCurrentIndex(1);
}

void GUIforDemo::on_radioButton_3_clicked()
{
	ui.stackedWidget->setCurrentIndex(2);
}

void GUIforDemo::on_pushButton_clicked()
{
	AddPointDialog dialog(this);
	dialog.exec();
}

void GUIforDemo::on_pushButton_5_clicked()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Image"), "c:/", tr("Image Files (*.png *.jpg *.bmp)"));
}

void GUIforDemo::on_pushButton_7_clicked()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Image"), "c:/", tr("Image Files (*.png *.jpg *.bmp)"));
}

void GUIforDemo::on_pushButton_10_clicked()
{
	AddSurfaceDialog dialog(this);
	dialog.exec();
}

void GUIforDemo::on_pushButton_13_clicked()
{
	tracer::test();
}
