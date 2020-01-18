//
//  AtomicIntRepresentable.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#if SWIFT_PACKAGE
import CCoroutine
#endif

@propertyWrapper
public struct AtomicIntRepresentable<T: RawRepresentable> where T.RawValue == Int {
    
    @usableFromInline var _value: Int
    
    @inlinable public init(wrappedValue value: T) {
        _value = value.rawValue
    }
    
    @inlinable public var wrappedValue: T {
        get { T(rawValue: _value)! }
        set {
            let value = OpaquePointer(UnsafeMutablePointer(&_value))
            let newValue = newValue.rawValue
            var oldValue: Int
            repeat { oldValue = _value }
                while __compare(value, &oldValue, newValue) == 0
        }
    }
    
}


