//
//  CoFutureError.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public enum CoFutureError: Error {
    case canceled, timeout
}
