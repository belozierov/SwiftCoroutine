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
    
    @usableFromInline private(set) var rawValue: Int
    
    public init(wrappedValue value: T) {
        rawValue = value.rawValue
    }
    
     public var wrappedValue: T {
        @inlinable get { T(rawValue: rawValue)! }
        set {
            let value = OpaquePointer(UnsafeMutablePointer(&rawValue))
            let newValue = newValue.rawValue
            var oldValue: Int
            repeat { oldValue = rawValue }
                while __compare(value, &oldValue, newValue) == 0
        }
    }
    
}


