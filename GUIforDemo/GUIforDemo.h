#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_GUIforDemo.h"

class GUIforDemo : public QMainWindow
{
	Q_OBJECT

public:
	GUIforDemo(QWidget *parent = Q_NULLPTR);

private:
	Ui::GUIforDemoClass ui;

private slots:
	void on_radioButton_clicked();
	void on_radioButton_2_clicked();
	void on_radioButton_3_clicked();
	void on_pushButton_clicked();
	void on_pushButton_5_clicked();
	void on_pushButton_7_clicked();
	void on_pushButton_10_clicked();
	void on_pushButton_13_clicked();
};
