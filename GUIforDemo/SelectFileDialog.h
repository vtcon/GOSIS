#pragma once

#include <QFileDialog>
#include "ui_SelectFileDialog.h"

class SelectFileDialog : public QFileDialog, public Ui::SelectFileDialog
{
	Q_OBJECT

public:
	SelectFileDialog(QWidget *parent = Q_NULLPTR);
	~SelectFileDialog();
};
