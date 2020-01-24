//
//  AtomicIntRepresentable.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

@propertyWrapper
public struct AtomicIntRepresentable<T: RawRepresentable> where T.RawValue == Int {
    
    @usableFromInline
    @AtomicInt private(set) var rawValue: Int
    
    public init(wrappedValue value: T) {
        rawValue = value.rawValue
    }
    
     public var wrappedValue: T {
        @inlinable get { T(rawValue: rawValue)! }
        set { rawValue = newValue.rawValue }
    }
    
}


