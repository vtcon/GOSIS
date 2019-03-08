#pragma once

#include <QDialog>
#include "ui_CloneConfigDialog.h"

class CloneConfigDialog : public QDialog, public Ui::CloneConfigDialog
{
	Q_OBJECT

public:
	CloneConfigDialog(QWidget *parent = Q_NULLPTR);
	~CloneConfigDialog();
};
