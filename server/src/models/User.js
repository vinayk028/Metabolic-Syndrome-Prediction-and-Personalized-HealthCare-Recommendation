const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
    firstName: {
        type: String,
        required: [true, 'First name is required'],
        trim: true,
        minlength: [2, 'First name must be at least 2 characters'],
        maxlength: [50, 'First name cannot exceed 50 characters']
    },
    lastName: {
        type: String,
        required: [true, 'Last name is required'],
        trim: true,
        minlength: [2, 'Last name must be at least 2 characters'],
        maxlength: [50, 'Last name cannot exceed 50 characters']
    },
    email: {
        type: String,
        required: [true, 'Email is required'],
        unique: true,
        lowercase: true,
        trim: true,
        match: [/^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$/, 'Please enter a valid email']
    },
    password: {
        type: String,
        required: [true, 'Password is required'],
        minlength: [6, 'Password must be at least 6 characters'],
        select: false
    },
    phone: {
        type: String,
        trim: true,
        default: ''
    },
    dateOfBirth: {
        type: Date,
        default: null
    },
    gender: {
        type: String,
        enum: ['Male', 'Female', 'Other', ''],
        default: ''
    },
    address: {
        type: String,
        trim: true,
        default: ''
    },
    profileImage: {
        type: String,
        default: ''
    },
    role: {
        type: String,
        enum: ['user', 'admin'],
        default: 'user'
    },
    isActive: {
        type: Boolean,
        default: true
    },
    lastLogin: {
        type: Date,
        default: null
    },
    assessmentHistory: [{
        date: {
            type: Date,
            default: Date.now
        },
        probability: Number,
        severity: Number,
        riskLevel: String,
        inputParameters: {
            age: Number,
            gender: String,
            fattyLiver: Boolean,
            hypertension: Boolean,
            diabetes: Boolean,
            systolicBP: Number,
            diastolicBP: Number,
            waistCircumference: Number,
            hdlCholesterol: Number,
            triglyceride: Number,
            fpg: Number
        },
        recommendations: {
            dietPlan: [String],
            avoidList: [String],
            exercisePlan: [String],
            yogaPoses: [String]
        }
    }]
}, {
    timestamps: true
});

// Keep only the latest 7 assessments
userSchema.pre('save', function(next) {
    if (this.assessmentHistory && this.assessmentHistory.length > 7) {
        // Sort by date descending, keep only latest 7
        this.assessmentHistory.sort((a, b) => new Date(b.date) - new Date(a.date));
        this.assessmentHistory = this.assessmentHistory.slice(0, 7);
    }
    next();
});

// Hash password before saving
userSchema.pre('save', async function(next) {
    if (!this.isModified('password')) {
        return next();
    }
    
    const salt = await bcrypt.genSalt(12);
    this.password = await bcrypt.hash(this.password, salt);
    next();
});

// Compare password method
userSchema.methods.comparePassword = async function(candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
};

// Get full name virtual
userSchema.virtual('fullName').get(function() {
    return `${this.firstName} ${this.lastName}`;
});

// Ensure virtuals are included in JSON output
userSchema.set('toJSON', { virtuals: true });
userSchema.set('toObject', { virtuals: true });

module.exports = mongoose.model('User', userSchema);
