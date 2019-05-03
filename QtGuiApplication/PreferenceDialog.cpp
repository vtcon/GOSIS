#include "PreferenceDialog.h"

#include "../test2/ProgramInterface.h"

tracer::PI_Preferences initialPref;

bool operator==(const tracer::PI_Preferences& lhs, const tracer::PI_Preferences& rhs)
{
	bool cond1 = lhs.ThreadsPerKernelLaunch == rhs.ThreadsPerKernelLaunch;
	bool cond2 = lhs.linearRayDensity == rhs.linearRayDensity;
	bool cond3 = lhs.rgbStandard == rhs.rgbStandard;
	bool cond4 = lhs.traceJobSize == rhs.traceJobSize;
	bool cond5 = lhs.renderJobSize == rhs.renderJobSize;
	bool cond6 = lhs.rawFormat == rhs.rawFormat;
	bool cond7 = lhs.projectionMethod == rhs.projectionMethod;
	bool cond8 = lhs.displayWindowSize == rhs.displayWindowSize;
	bool cond9 = lhs.primaryWavelengthR == rhs.primaryWavelengthR;
	bool cond10 = lhs.primaryWavelengthG == rhs.primaryWavelengthG;
	bool cond11 = lhs.primaryWavelengthB == rhs.primaryWavelengthB;
	bool cond12 = lhs.maxParallelThread == rhs.maxParallelThread;
	bool cond13 = lhs.performanceTestRepetition == rhs.performanceTestRepetition;
	return cond1 && cond2 && cond3 && cond4 && cond5
		&& cond6 && cond7 && cond8 && cond9 && cond10
		&& cond11 && cond12 && cond13;
}

bool operator!=(const tracer::PI_Preferences& lhs, const tracer::PI_Preferences& rhs)
{
	return !(lhs == rhs);
}

PreferenceDialog::PreferenceDialog(QWidget *parent)
	: QDialog(parent)
{
	setupUi(this);
	tracer::PI_Preferences loadPref;
	tracer::getPreferences(loadPref);
	initialPref = loadPref;
	switch (loadPref.ThreadsPerKernelLaunch)
	{
	case 128:
		comboThreadCount->setCurrentIndex(4);
		break;
	case 64:
		comboThreadCount->setCurrentIndex(3);
		break;
	case 8:
		comboThreadCount->setCurrentIndex(2);
		break;
	case 16:
		comboThreadCount->setCurrentIndex(1);
		break;
	case 32:
	default:
		comboThreadCount->setCurrentIndex(0);
		break;
	}
	lineRayGeneration->setText(QString::number(loadPref.linearRayDensity));
	lineTraceSize->setText(QString::number(loadPref.traceJobSize));
	lineRenderSize->setText(QString::number(loadPref.renderJobSize));
	switch (loadPref.rgbStandard)
	{
	case PI_SRGB:
		comboRGB->setCurrentIndex(1);
		break;
	case PI_ADOBERGB:
	default:
		comboRGB->setCurrentIndex(0);
		break;
	}
	switch (loadPref.rawFormat)
	{
	case PI_LMS:
		comboRaw->setCurrentIndex(1);
		break;
	case PI_XYZ:
	default:
		comboRaw->setCurrentIndex(0);
		break;
	}
	switch (loadPref.projectionMethod)
	{
	case PI_PROJECTION_NONE:
		comboProjection->setCurrentIndex(1);
		break;
	case PI_PROJECTION_ALONGZ:
		comboProjection->setCurrentIndex(2);
		break;
	case PI_PROJECTION_PLATE_CARREE:
	default:
		comboProjection->setCurrentIndex(0);
		break;
	}
	linePreviewSize->setText(QString::number(loadPref.displayWindowSize));
	lineWavelengthR->setText(QString::number(loadPref.primaryWavelengthR));
	lineWavelengthG->setText(QString::number(loadPref.primaryWavelengthG));
	lineWavelengthB->setText(QString::number(loadPref.primaryWavelengthB));
	lineCPUThread->setText(QString::number(loadPref.maxParallelThread));
	lineTestKernelRepetition->setText(QString::number(loadPref.performanceTestRepetition));
}

PreferenceDialog::~PreferenceDialog()
{
}

void PreferenceDialog::on_pushOK_clicked()
{
	tracer::PI_Preferences newPref;
	//read in new pref and scan for error

	int valCPUThread = lineCPUThread->text().toInt();
	valCPUThread = ((valCPUThread <= 1) ? 1 : valCPUThread) >= 128 ? 128 : valCPUThread;
	lineCPUThread->setText(QString::number(valCPUThread));
	newPref.maxParallelThread = valCPUThread;

	switch (comboThreadCount->currentIndex())
	{
	case 4:
		newPref.ThreadsPerKernelLaunch = 128;
		break;
	case 3:
		newPref.ThreadsPerKernelLaunch = 64;
		break;
	case 2:
		newPref.ThreadsPerKernelLaunch = 8;
		break;
	case 1:
		newPref.ThreadsPerKernelLaunch = 16;
		break;
	case 0:
	default:
		newPref.ThreadsPerKernelLaunch = 32;
		break;
	}

	int valRayGen = lineRayGeneration->text().toInt();
	valRayGen = ((valRayGen <= 1) ? 1 : valRayGen) >= 200 ? 200 : valRayGen;
	lineRayGeneration->setText(QString::number(valRayGen));
	newPref.linearRayDensity = valRayGen;

	int valTraceSize = lineTraceSize->text().toInt();
	valTraceSize = ((valTraceSize <= 1) ? 1 : valTraceSize) >= 100 ? 100 : valTraceSize;
	lineTraceSize->setText(QString::number(valTraceSize));
	newPref.traceJobSize = valTraceSize;

	int valRenderSize = lineRenderSize->text().toInt();
	valRenderSize = ((valRenderSize <= 1) ? 1 : valRenderSize) >= 100 ? 100 : valRenderSize;
	lineRenderSize->setText(QString::number(valRenderSize));
	newPref.renderJobSize = valRenderSize;

	switch (comboRGB->currentIndex())
	{
	case 1:
		newPref.rgbStandard = PI_SRGB;
		break;
	case 0:
	default:
		newPref.rgbStandard = PI_ADOBERGB;
		break;
	}

	switch (comboRaw->currentIndex())
	{
	case 1:
		newPref.rawFormat = PI_LMS;
		break;
	case 0:
	default:
		newPref.rawFormat = PI_XYZ;
		break;
	}

	switch (comboProjection->currentIndex())
	{
	case 2:
		newPref.projectionMethod = PI_PROJECTION_ALONGZ;
		break;
	case 1:
		newPref.projectionMethod = PI_PROJECTION_NONE;
		break;
	case 0:
	default:
		newPref.projectionMethod = PI_PROJECTION_PLATE_CARREE;
		break;
	}

	int valPreviewSize = linePreviewSize->text().toInt();
	valPreviewSize = ((valPreviewSize <= 50) ? 50 : valPreviewSize) >= 2048 ? 2048 : valPreviewSize;
	linePreviewSize->setText(QString::number(valPreviewSize));
	newPref.displayWindowSize = valPreviewSize;

	float valWavelengthR = ((float)round(10 * lineWavelengthR->text().toFloat())) / 10.0;
	valWavelengthR = ((valWavelengthR <= 610) ? 610 : valWavelengthR) >= 630 ? 630 : valWavelengthR;
	lineWavelengthR->setText(QString::number(valWavelengthR));
	newPref.primaryWavelengthR = valWavelengthR;

	float valWavelengthG = ((float)round(10 * lineWavelengthG->text().toFloat())) / 10.0;
	valWavelengthG = ((valWavelengthG <= 520) ? 520 : valWavelengthG) >= 540 ? 540 : valWavelengthG;
	lineWavelengthG->setText(QString::number(valWavelengthG));
	newPref.primaryWavelengthG = valWavelengthG;

	float valWavelengthB = ((float)round(10 * lineWavelengthB->text().toFloat())) / 10.0;
	valWavelengthB = ((valWavelengthB <= 460) ? 460 : valWavelengthB) >= 475 ? 475 : valWavelengthB;
	lineWavelengthB->setText(QString::number(valWavelengthB));
	newPref.primaryWavelengthB = valWavelengthB;

	int testKernelRepetition = lineTestKernelRepetition->text().toInt();
	testKernelRepetition = ((testKernelRepetition <= 1) ? 1 : testKernelRepetition) >= 2000000000 ? 2000000000 : testKernelRepetition;
	lineTestKernelRepetition->setText(QString::number(testKernelRepetition));
	newPref.performanceTestRepetition = testKernelRepetition;

	//if the same as initial pref, just accept
	//else set new pref and accept
	if (newPref != initialPref)
	{
		if (tracer::setPreferences(newPref).code != PI_OK)
		{
			tracer::defaultPreference();
		}
	}
	accept();
}

void PreferenceDialog::on_pushDefault_clicked()
{
	tracer::defaultPreference();
	tracer::PI_Preferences loadPref;
	tracer::getPreferences(loadPref);
	switch (loadPref.ThreadsPerKernelLaunch)
	{
	case 128:
		comboThreadCount->setCurrentIndex(4);
		break;
	case 64:
		comboThreadCount->setCurrentIndex(3);
		break;
	case 8:
		comboThreadCount->setCurrentIndex(2);
		break;
	case 16:
		comboThreadCount->setCurrentIndex(1);
		break;
	case 32:
	default:
		comboThreadCount->setCurrentIndex(0);
		break;
	}
	lineRayGeneration->setText(QString::number(loadPref.linearRayDensity));
	lineTraceSize->setText(QString::number(loadPref.traceJobSize));
	lineRenderSize->setText(QString::number(loadPref.renderJobSize));
	switch (loadPref.rgbStandard)
	{
	case PI_SRGB:
		comboRGB->setCurrentIndex(1);
		break;
	case PI_ADOBERGB:
	default:
		comboRGB->setCurrentIndex(0);
		break;
	}
	switch (loadPref.rawFormat)
	{
	case PI_LMS:
		comboRaw->setCurrentIndex(1);
		break;
	case PI_XYZ:
	default:
		comboRaw->setCurrentIndex(0);
		break;
	}
	switch (loadPref.projectionMethod)
	{
	case PI_PROJECTION_NONE:
		comboProjection->setCurrentIndex(1);
		break;
	case PI_PROJECTION_ALONGZ:
		comboProjection->setCurrentIndex(2);
		break;
	case PI_PROJECTION_PLATE_CARREE:
	default:
		comboProjection->setCurrentIndex(0);
		break;
	}
	linePreviewSize->setText(QString::number(loadPref.displayWindowSize));
	lineWavelengthR->setText(QString::number(loadPref.primaryWavelengthR));
	lineWavelengthG->setText(QString::number(loadPref.primaryWavelengthG));
	lineWavelengthB->setText(QString::number(loadPref.primaryWavelengthB));
	lineCPUThread->setText(QString::number(loadPref.maxParallelThread));
	lineTestKernelRepetition->setText(QString::number(loadPref.performanceTestRepetition));
}