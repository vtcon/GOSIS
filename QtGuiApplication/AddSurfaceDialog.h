#pragma once

#include <QDialog>
#include "ui_AddSurfaceDialog.h"

class AddSurfaceDialog : public QDialog, public Ui::AddSurfaceDialog
{
	Q_OBJECT

public:
	AddSurfaceDialog(QWidget *parent = Q_NULLPTR);
	~AddSurfaceDialog();

private slots:
	void on_pushAddSurface_clicked();
	void on_comboBox_currentIndexChanged();
	void on_pushSelectApoPath_clicked();
};
