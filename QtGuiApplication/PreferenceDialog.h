#pragma once

#include <QDialog>
#include "ui_PreferenceDialog.h"

class PreferenceDialog : public QDialog, public Ui::PreferenceDialog
{
	Q_OBJECT

public:
	PreferenceDialog(QWidget *parent = Q_NULLPTR);
	~PreferenceDialog();

private slots:

	void on_pushOK_clicked();
	void on_pushDefault_clicked();
};
